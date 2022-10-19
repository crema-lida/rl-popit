import numpy as np
import torch
import torch.nn.functional as f
from torch.nn.utils import clip_grad_norm_
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import os, time, pickle
from tqdm import tqdm, trange

import model
from game import Env
from player import Player
import utils


def train_with_ppo(epochs=10000, minibatch_size=1024, eps=0.2):
    if os.path.exists(f'{MODEL_DIR}/checkpoint'):
        with open(f'{MODEL_DIR}/checkpoint', 'rb') as checkpoint:
            model_idx = pickle.load(checkpoint)
    else:
        model_idx = 1

    for ep in trange(epochs, ncols=80, desc='Progress'):
        # save the latest model for opponent to use
        if ep % 100 == 0:
            model_path = f'{MODEL_DIR}/opp/model-{model_idx}'
            torch.save(agent.model.state_dict(), model_path)
            with open(f'{MODEL_DIR}/checkpoint', 'wb') as checkpoint:
                pickle.dump(model_idx, checkpoint)
            tqdm.write(f'New model state saved to {model_path}')
            if os.path.exists(summary_dir := f'{SUMMARY_DIR}/win_rate_model-{model_idx}'):
                os.system(f'rm {summary_dir}/*')
            model_idx = model_idx + 1 if model_idx < 30 else 1

        tqdm.write(f'Epoch {ep}'.center(40) + '\n' + 'Opponent     Win Rate (%)'.center(40))

        # sample (10 * 2 * batch_size) games by playing with older models
        for _ in trange(10, ncols=80, leave=False, desc='Self Play'):
            # randomly select a model from history
            model_name = np.random.choice(os.listdir(f'{MODEL_DIR}/opp'))
            opponent.load_state_dict(torch.load(f'{MODEL_DIR}/opp/{model_name}'))

            wins = 0

            for first_move in range(2):
                player_idx = first_move
                state, done, reward = env.reset()
                while np.any(~done):
                    if player_idx == 1:
                        state[:, [0, 1]] = state[:, [1, 0]]
                    policy, action = Player.select_action(cnn[player_idx], state)
                    state_new, reward, done_new = env.step(state.copy(), action, player_idx)

                    if player_idx == 0:
                        agent.remember(state, policy, action, done)
                    elif env.graphics:
                        policy, _ = Player.select_action(agent.model, state_new)
                        env.paint_canvas(policy)
                    if reward is not None:
                        agent.assign_reward(reward)
                    state, done = state_new, done_new
                    env.render(state[:4])
                    player_idx = 1 - player_idx

                wins += 0.5 * (reward.sum() + env.batch_size)

            win_rate = wins / (2 * env.batch_size) * 100
            if int(model_name[6:]) % 5 == 1:
                writer.add_scalars('win_rate', {model_name: win_rate}, ep)
            tqdm.write(f'{model_name}     {win_rate: .2f}'.center(40))

        info = agent.learn(minibatch_size, eps)
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
                    action = np.full(env.batch_size, -1)  # (N,)
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

                        num = min(env.batch_size, 4)
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

                    action = np.full(env.batch_size, -1)
                    action[~done] = np.concatenate([
                        choice(p) for p in policy
                    ])

                state, done, rewards = env.step(env.state, action)
                player = 1 - player

            if player == 0: rewards = -rewards
            wins += 0.5 * (rewards.sum() + env.batch_size)

        for stage in agent.policy_network:
            for param in stage.parameters():
                clip_grad_norm_(param, max_norm=10, error_if_nonfinite=True)
        policy_optim.step()
        policy_optim.zero_grad()

        win_rate = wins / (2 * env.batch_size)
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
        action = np.full(env.batch_size, -1)
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
    env.batch_size = 1
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
            action = np.full(env.batch_size, -1)  # (N,)
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

                num = min(env.batch_size, 4)
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
    MODEL_DIR = 'model-conv3'
    SUMMARY_DIR = 'runs/' + time.asctime().replace(' ', '-')  # '/tf_logs'
    # os.system('rm -rf /tf_logs/*')
    writer = SummaryWriter(SUMMARY_DIR)

    env = Env(graphics=False, fps=1, batch_size=128)
    utils.device = torch.device('cuda')
    in_features = env.state.shape[1]
    agent = Player(model.CNN(in_features, num_blocks=3).to(utils.device),
                   lr=2e-4, weight_decay=1e-4)
    opponent = model.CNN(in_features, num_blocks=3).to(utils.device)
    cnn = [agent.model, opponent]

    os.makedirs(f'{MODEL_DIR}/opp', exist_ok=True)
    if os.path.exists(f'{MODEL_DIR}/agent'):
        agent.model.load_state_dict(torch.load(f'{MODEL_DIR}/agent'))
    if os.path.exists(f'{MODEL_DIR}/policy_optim'):
        agent.policy_optim.load_state_dict(torch.load(f'{MODEL_DIR}/policy_optim'))
    if os.path.exists(f'{MODEL_DIR}/value_optim'):
        agent.value_optim.load_state_dict(torch.load(f'{MODEL_DIR}/value_optim'))

    train_with_ppo(minibatch_size=2048, eps=0.1)
    # train_with_mcts()
    # play_with_mcts()

    env.close()
