import numpy as np
import torch
import torch.nn.functional as f
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
import os, time
from tqdm import tqdm

import network
from game import Env
from agent import Agent
import utils


def train_with_ppo(epochs=20000, ep_start=0, sample_loops=10, drop_threshold=85):
    ema_win_rate = {}
    with tqdm(total=epochs, ncols=80, desc='Progress') as tbar_main:
        tbar_main.update(ep_start)
        for ep in range(ep_start, epochs):
            tbar_main.set_postfix_str(f'lr={agent.scheduler.get_last_lr()[0]:.2e}')
            tbar_main.update(1)
            tqdm.write(f'Epoch {ep}'.center(45) + '\n' + 'Opponent     Win Rate (%)'.center(45))
            if (ep <= 1000 and ep % 100 == 0 or ep > 1000 and ep % 500 == 0) and \
                    np.all(np.array(list(ema_win_rate.values())) > 55):
                agent.save_as_opponent(MODEL_DIR, ep)

            # sample (loops * batch_size) games by playing with older models
            agent.model.eval()
            with tqdm(total=sample_loops, ncols=80, leave=False, desc='Self Play') as tbar:
                for _ in range(sample_loops // 2):
                    dirlist = os.listdir(f'{MODEL_DIR}/opp')
                    model_name = np.random.choice(dirlist)  # randomly select a model from history
                    opponent.load_state_dict(torch.load(f'{MODEL_DIR}/opp/{model_name}'))
                    # opponent.load_state_dict(torch.load('resnet3/checkpoint')['model'])
                    wins = 0
                    for first_move in range(2):
                        tbar.update(1)
                        player_idx = first_move
                        state, reward, done = env.reset()
                        while np.any(~done):
                            if player_idx == 1:
                                state[:, [0, 1]] = state[:, [1, 0]]
                            policy, action = Agent.choose_action(cnn[player_idx], state)
                            state_new, reward, done_new = env.step(state.copy(), action, player_idx)

                            if player_idx == 0:
                                agent.remember(state, policy, action, done)
                            elif env.graphics:
                                policy, _ = Agent.choose_action(agent.model, state_new)
                                env.paint_canvas(policy)

                            state, done = state_new, done_new
                            env.render()
                            player_idx = 1 - player_idx

                        agent.assign_reward(reward)
                        wins += 0.5 * (reward.sum() + env.num_envs)

                    win_rate = wins / (2 * env.num_envs) * 100
                    if model_name not in ema_win_rate:
                        ema_win_rate[model_name] = win_rate
                    else:
                        ema_win_rate[model_name] = ema_win_rate[model_name] * 0.9 + win_rate * 0.1
                    tqdm.write(f'{model_name}     {win_rate: .2f}'.center(40))
                    if ema_win_rate[model_name] > drop_threshold:
                        # remove the old model, and save the latest for opponent to use
                        os.remove(f'{MODEL_DIR}/opp/{model_name}')
                        agent.save_as_opponent(MODEL_DIR, ep)
                        del ema_win_rate[model_name]

            agent.model.train()
            info = agent.learn()
            agent.scheduler.step()
            for tag, data in info.items():
                if tag == 'value_loss':
                    writer.add_scalar(tag, torch.concat(data).mean(), ep)
                else:
                    writer.add_histogram(f'metrics/{tag}', torch.concat(data), ep)
            writer.add_scalars('win_rate', ema_win_rate, ep)
            agent.save_checkpoint(MODEL_DIR, ep)


def train_with_mcts(epochs=10000):
    cross_entropy = torch.nn.CrossEntropyLoss()
    start = time.time()
    for ep in range(epochs + 1):
        opponent.load_state_dict(agent.model.state_dict())
        wins = 0

        for first_move in range(2):
            player = first_move
            state, done, rewards = env.reset()
            while np.any(~done):
                state[:, [0, 1]] = state[:, [1, 0]]

                if player == 0:
                    idx = np.arange(len(state[~done]))
                    _state = from_numpy(augment_data(state[~done]), device)  # (n, 2, 6, 6)
                    _mask = _state[:, 1].reshape(-1, 36) != 0
                    _state = _state - torch.mean(_state, dim=(2, 3), keepdim=True)  # zero-center
                    out = cnn[player](_state)
                    _policy = cnn[player].policy_head(out, _mask)  # (n, 36)
                    policy = _policy[idx].detach().cpu().numpy()
                    policy[_mask[idx].cpu().numpy()] = -np.inf

                    action_value = np.zeros_like(policy)  # (n, 36)
                    n = np.zeros_like(policy)  # (n, 36)
                    action = np.full(env.num_envs, -1)  # (N,)
                    mask = _mask[idx].cpu()

                    turns = env.num_turns

                    for i in range(240):
                        q = action_value / (1e-5 + n)  # (n, 36)
                        score = q + 10 * policy / (n + 1)  # (n, 36)
                        selection = i if i < 36 else np.argmax(score, axis=1)
                        action[~done] = selection
                        action_value[idx, selection] += rollout(state.copy(), action)[~done]
                        n[idx, selection] += 1
                        if i < 36:
                            action_value[mask] = 0
                            n[mask] = 0

                        num = min(env.num_envs, 4)
                        cmap, p_value, q_value, n_value = [np.zeros((num, 36)) for _ in range(4)]
                        sel = ~done[:4]
                        len_sel = len(np.where(sel)[0])
                        exp_cmap = np.exp(score[:len_sel])
                        _cmap = exp_cmap / np.sum(exp_cmap, axis=1, keepdims=True)
                        cmap[sel] = _cmap / np.max(_cmap, axis=1, keepdims=True)
                        env.paint_canvas(cmap)

                        p_value[sel] = policy[:len_sel]
                        q_value[sel] = q[:len_sel]
                        n_value[sel] = n[:len_sel]
                        env.render_text(p_value, q_value, n_value)
                        env.render(env.state[:4])
                        env.num_turns = turns

                    n = from_numpy(augment_data(n.reshape((-1, 1, 6, 6))).reshape((-1, 36)), device)
                    target_p = f.softmax(n, dim=1)
                    loss = cross_entropy(_policy, target_p)
                    loss.backward()

                    action[~done] = np.concatenate([
                        choice(p.cpu().numpy()) for p in target_p[idx]
                    ])
                else:
                    _state = from_numpy(state[~done], device)
                    _mask = _state[:, 1].reshape(-1, 36) != 0
                    out = cnn[player](_state)
                    _policy = cnn[player].policy_head(out, _mask)  # (n, 36)
                    policy = _policy.detach().cpu().numpy()

                    action = np.full(env.num_envs, -1)
                    action[~done] = np.concatenate([
                        choice(p) for p in policy
                    ])

                state, done, rewards = env.step(env.state, action)
                player = 1 - player

            if player == 0: rewards = -rewards
            wins += 0.5 * (rewards.sum() + env.num_envs)

        for stage in agent.policy_network:
            for param in stage.parameters():
                clip_grad_norm_(param, max_norm=10, error_if_nonfinite=True)
        policy_optim.step()
        policy_optim.zero_grad()

        win_rate = wins / (2 * env.num_envs)
        print(f'Epoch {ep} | Win Rate: {win_rate * 100:.2f} % | '
              f'elapsed: {time.time() - start:.2f} s')
        torch.save(agent.state_dict(), f'{MODEL_DIR}/agent')


def play_with_mcts(policy_only=False, max_searches=180, policy_weight=1.0):
    assert env.graphics, 'Interactive mode requires graphics=True.'
    utils.device = torch.device('cpu')
    for nn in cnn:
        nn.to(utils.device)
        nn.to(utils.device)
    env.num_envs = 1
    env.mode = 'interactive'
    agent.model.eval()
    player_idx = np.random.randint(2)
    state, reward, done = env.reset()
    while np.any(~done):
        if player_idx == 1:
            state[0, [0, 1]] = state[0, [1, 0]]
            action = env.wait()
        else:
            policy, _ = agent.choose_action(agent.model, state)
            if policy_only:
                n = policy
                env.paint_canvas(policy)
            else:
                mask = state[0, 1].reshape(-1, 36) != 0
                policy[mask] = -np.inf
                q = np.zeros_like(policy)  # (1, 36) the action value
                n = np.zeros_like(policy)  # (1, 36)
                for i in range(max_searches):
                    score = q + policy_weight * policy / (n + 1)  # (1, 36)
                    sel = np.argmax(score, axis=1)
                    q[0, sel] = 0.9 * q[0, sel] + 0.1 * agent.rollout(state.copy(), sel, env.num_turns)
                    n[0, sel] += 1
                    exp_cmap = np.exp(score)
                    cmap = exp_cmap / exp_cmap.sum()
                    env.paint_canvas(cmap)
                    env.render_text(policy, q, n)
                    env.render()
            action = np.argmax(n, axis=-1)

        state, reward, done = env.step(state.copy(), action, player_idx)
        env.render()
        if np.all(done): time.sleep(1)
        player_idx = 1 - player_idx


if __name__ == '__main__':
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    MODEL_DIR = 'resnet3'
    SUMMARY_DIR = 'runs/' + time.asctime().replace(' ', '-').replace(':', '.')
    # SUMMARY_DIR = '/tf_logs'
    # os.system('rm -rf /tf_logs/*')
    writer = SummaryWriter(SUMMARY_DIR)
    utils.device = torch.device('cuda')

    env = Env(graphics=True, fps=3, num_envs=64)
    in_features = env.state.shape[1]
    agent = Agent(network.ResNet(in_features, num_blocks=3).to(utils.device),
                  minibatch_size=2048, clip=0.1, entropy_coeff=0.01, max_norm=0.5,
                  lr=5e-4, weight_decay=1e-4, eps=1e-5,
                  t_max=10000, min_lr=5e-6)
    opponent = network.ResNet(in_features, num_blocks=3).to(utils.device).eval()
    cnn = [agent.model, opponent]

    os.makedirs(f'{MODEL_DIR}/opp', exist_ok=True)
    if os.path.exists(f'{MODEL_DIR}/checkpoint'):
        checkpoint = torch.load(f'{MODEL_DIR}/checkpoint')
        agent.model.load_state_dict(checkpoint['model'])
        agent.policy_optim.load_state_dict(checkpoint['policy_optim'])
        agent.value_optim.load_state_dict(checkpoint['value_optim'])
        agent.scheduler.load_state_dict(checkpoint['scheduler'])
        ep = checkpoint['epoch']
    else:
        ep = 0

    # train_with_ppo(epochs=20000, ep_start=ep, sample_loops=20, drop_threshold=85)
    play_with_mcts(policy_only=True, max_searches=180, policy_weight=1.0)
    # train_with_mcts()

    env.close()
