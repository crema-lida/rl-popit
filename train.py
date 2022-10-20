import numpy as np
import torch
import torch.nn.functional as f
from torch.nn.utils import clip_grad_norm_
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import os, time
from tqdm import tqdm, trange

import network
from game import Env
from agent import Agent
import utils


def train_with_ppo(epochs=10000, sample_loops=10, max_saved_models=10, drop_threshold=95):
    # while (idx := len(os.listdir(f'{MODEL_DIR}/opp'))) < max_saved_models:
    #     model_path = f'{MODEL_DIR}/opp/model-{idx + 1}'
    #     torch.save(agent.model.state_dict(), model_path)
    model_idx = 1
    model_path = f'{MODEL_DIR}/opp/model-{model_idx}'
    torch.save(agent.model.state_dict(), model_path)

    for ep in trange(epochs, ncols=80, desc='Progress'):
        tqdm.write(f'Epoch {ep}'.center(45) + '\n' + 'Opponent     Win Rate (%)'.center(45))

        # sample (loops * batch_size) games by playing with older models
        with tqdm(total=sample_loops, ncols=80, leave=False, desc='Self Play') as tbar:
            for _ in range(sample_loops // 2):
                dirlist = os.listdir(f'{MODEL_DIR}/opp')
                model_name = np.random.choice(dirlist)  # randomly select a model from history
                opponent.load_state_dict(torch.load(f'{MODEL_DIR}/opp/{model_name}'))
                wins = 0
                for first_move in range(2):
                    tbar.update(1)
                    player_idx = first_move
                    state, done, reward = env.reset()
                    while np.any(~done):
                        if player_idx == 1:
                            state[:, [0, 1]] = state[:, [1, 0]]
                        policy, action = Agent.select_action(cnn[player_idx], state)
                        state_new, reward, done_new = env.step(state.copy(), action, player_idx)

                        if player_idx == 0:
                            agent.remember(state, policy, action, done)
                        elif env.graphics:
                            policy, _ = Agent.select_action(agent.model, state_new)
                            env.paint_canvas(policy)

                        state, done = state_new, done_new
                        env.render(state[:4])
                        player_idx = 1 - player_idx

                    agent.assign_reward(reward)
                    wins += 0.5 * (reward.sum() + env.num_envs)

                win_rate = wins / (2 * env.num_envs) * 100
                writer.add_scalars('win_rate', {model_name: win_rate}, ep)
                tqdm.write(f'{model_name}     {win_rate: .2f}'.center(40))
                # if win_rate > drop_threshold:
                if (ep % 100 == 0 and ep > 1 and model_idx < 5) or (model_idx == 1 and win_rate > 70):
                    model_idx += 1
                    model_path = f'{MODEL_DIR}/opp/{model_idx}'
                    # save the latest model for opponent to use
                    # model_path = f'{MODEL_DIR}/opp/{model_name}'
                    torch.save(agent.model.state_dict(), model_path)
                    tqdm.write(f'New model state saved to {model_path}')
                    if os.path.exists(summary_dir := f'{SUMMARY_DIR}/win_rate_{model_name}'):
                        os.system(f'rm {summary_dir}/*')

        info = agent.learn()
        for tag, data in info.items():
            writer.add_histogram(tag, torch.concat(data), ep)
        agent.save_state_dict(MODEL_DIR)


def train_with_mcts(epochs=10000):
    cross_entropy = torch.nn.CrossEntropyLoss()
    start = time.time()
    for ep in range(epochs + 1):
        opponent.load_state_dict(agent.state_dict())
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

                    turns = env.turns

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
                        env.turns = turns

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


def rollout(state, action):
    state, done, rewards = env.step(state, action)
    player = 1
    while np.any(~done):
        state[:, [0, 1]] = state[:, [1, 0]]
        _state = utils.from_numpy(state[~done])
        mask = _state[:, 1].reshape(-1, 36) != 0
        _state = _state - torch.mean(_state, dim=(2, 3), keepdim=True)  # zero-center
        out = cnn[player](_state)
        policy = cnn[player].policy_head(out, mask).detach().cpu()
        action = np.full(env.num_envs, -1)
        action[~done] = Categorical(policy).sample()

        state, done, rewards = env.step(state, action)
        player = 1 - player
    return rewards if player == 1 else -rewards


def play_with_mcts():
    assert env.graphics, 'Interactive mode requires graphics=True.'
    utils.device = torch.device('cpu')
    for nn in cnn:
        nn.to(utils.device)
        nn.to(utils.device)
    env.num_envs = 1
    env.mode = 'interactive'
    player = np.random.randint(2)
    state, done, rewards = env.reset()
    while np.any(~done):
        state[:, [0, 1]] = state[:, [1, 0]]

        if player == 0:
            idx = np.arange(len(state[~done]))
            _state = utils.from_numpy(utils.augment_data(state[~done]))  # (n, 2, 6, 6)
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

            turns = env.turns

            for i in range(180):
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
                env.turns = turns

            n = utils.from_numpy(utils.augment_data(n.reshape((-1, 1, 6, 6))).reshape((-1, 36)))
            action[~done] = np.argmax(n[0])
        else:
            action = env.wait()

        state, done, rewards = env.step(env.state, action)
        if player == 0:
            env.render(env.state[:4])
        else:
            env.render(env.state[:4, [1, 0]])
        player = 1 - player


if __name__ == '__main__':
    MODEL_DIR = 'model-conv3-demo'
    SUMMARY_DIR = 'runs/' + time.asctime().replace(' ', '-')  # '/tf_logs'
    # os.system('rm -rf /tf_logs/*')
    writer = SummaryWriter(SUMMARY_DIR)

    env = Env(graphics=False, fps=3, num_envs=128)
    utils.device = torch.device('cuda')
    in_features = env.state.shape[1]
    agent = Agent(network.CNN(in_features, num_blocks=3).to(utils.device),
                  minibatch_size=2048, clip=0.1, entropy_coeff=0.01, max_norm=1,
                  lr=2e-3, weight_decay=1e-4, eps=1e-5)
    opponent = network.CNN(in_features, num_blocks=3).to(utils.device)
    cnn = [agent.model, opponent]

    os.makedirs(f'{MODEL_DIR}/opp', exist_ok=True)
    if os.path.exists(f'{MODEL_DIR}/agent'):
        agent.model.load_state_dict(torch.load(f'{MODEL_DIR}/agent'))
    if os.path.exists(f'{MODEL_DIR}/policy_optim'):
        agent.policy_optim.load_state_dict(torch.load(f'{MODEL_DIR}/policy_optim'))
    if os.path.exists(f'{MODEL_DIR}/value_optim'):
        agent.value_optim.load_state_dict(torch.load(f'{MODEL_DIR}/value_optim'))

    train_with_ppo(epochs=10000, sample_loops=20,
                   max_saved_models=10, drop_threshold=95)
    # train_with_mcts()
    # play_with_mcts()

    env.close()
