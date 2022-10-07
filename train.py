import numpy as np
import torch
import torch.nn.functional as f
from torch.nn.utils import clip_grad_norm_
import os, random
from time import time

from game import Env
from network import Network
from utils import SIZE, from_numpy, choice, augment_data


def train_with_policy_network(epochs=10000):
    model_idx = 7
    opp_model = {}
    for i in range(1, 21):
        model_path = f'model/opp_{i}'
        if os.path.exists(model_path):
            opp_model[model_path] = torch.load(model_path)

    start = time()
    for ep in range(epochs + 1):

        if ep % 20 == 0:
            model_path = f'model/opp_{model_idx}'
            opp_model[model_path] = agent.state_dict()
            torch.save(opp_model[model_path], model_path)
            model_idx = model_idx + 1 if model_idx < 20 else 1

            for net in agent.policy_network:
                clip_grad_norm_(net.parameters(), max_norm=0.1, error_if_nonfinite=True)
                # for name, param in net.named_parameters():
                #     print(name, param.grad.max().item())
            policy_optim.step()
            policy_optim.zero_grad()

            model_state = agent.state_dict()
            torch.save(model_state, 'model/agent')
            print(f'---------New model state saved to {model_path}---------')

        model_path = random.sample(list(opp_model.keys()), 1)[0]
        opponent.load_state_dict(opp_model[model_path])

        wins = 0

        # Agent makes the first move in the first batch,
        # then opponent makes the first move in succession.
        for first_move in range(2):
            player = first_move
            history = {'s': [], 'a': [], 'done': []}  # keeps a record of states and actions
            state, done, rewards = env.reset()

            while np.any(~done):
                state[:, [0, 1]] = state[:, [1, 0]]
                state[:, 2] = np.where(state[:, 0] == SIZE, 1, 0)  # positions full of agent's pieces
                state[:, 3] = np.where(state[:, 1] == SIZE, 1, 0)  # positions full of opponent's pieces

                state = from_numpy(state[~done], device)  # (n, 4, 6, 6)
                mask = state[:, 1].reshape(-1, 36) != 0
                out = cnn[player](state)
                policy = cnn[player].forward_policy_head(out, mask).detach().cpu().numpy()  # (n, 36)
                action = np.full(env.batch_size, -1)  # (N,)
                action[~done] = np.concatenate([
                    choice(p) for p in policy
                ])

                if player == 0:
                    history['s'].append(env.state[~done])
                    history['a'].append(action[~done])
                    history['done'].append(done)

                    if env.graphics:
                        num = min(env.batch_size, 4)
                        p = np.zeros((num, 36))
                        sel = ~done[:4]
                        len_sel = len(np.where(sel)[0])
                        p[sel] = policy[:len_sel] / policy[:len_sel].max(axis=1, keepdims=True)
                        env.paint_canvas(p)

                state, done, rewards = env.step(env.state, action)

                if player == 0:
                    env.render(env.state[:4])
                else:
                    env.render(env.state[:4, [1, 0]])
                player = 1 - player

            if player == 0: rewards = -rewards
            wins += 0.5 * (rewards.sum() + env.batch_size)
            rewards = from_numpy(rewards, device)
            for state, action, done in zip(*history.values()):
                size = len(action)
                act = np.full((size, 36), False)
                act[np.arange(size), action] = True

                action = torch.from_numpy(augment_data(act.reshape((-1, 1, 6, 6))).reshape((-1, 36)))
                state = from_numpy(augment_data(state), device)
                mask = state[:, 1].reshape(-1, 36) != 0
                out = agent(state)
                policy = agent.forward_policy_head(out, mask)
                torch.sum(
                    -torch.log(policy[action].clip(min=1e-8)) * rewards[~done].repeat(6)
                ).backward()

                # state_value = agent.forward_value_head(out.detach())
                # value_loss = f.mse_loss(state_value, rewards[~done].unsqueeze(1))
                # value_loss.backward()
                # value_optim.step()
                # value_optim.zero_grad()

        win_rate = wins / (2 * env.batch_size)
        print(f'Epoch {ep} | {model_path[6:]} | Win Rate: {win_rate * 100:.2f} % | '
              f'elapsed: {time() - start:.2f} s')


def train_with_mcts(epochs=10000):
    cross_entropy = torch.nn.CrossEntropyLoss()
    start = time()
    for ep in range(epochs + 1):
        opponent.load_state_dict(agent.state_dict())
        wins = 0

        for first_move in range(2):
            player = first_move
            state, done, rewards = env.reset()
            while np.any(~done):
                state[:, [0, 1]] = state[:, [1, 0]]
                state[:, 2] = np.where(state[:, 0] == SIZE, 1, 0)  # positions full of agent's pieces
                state[:, 3] = np.where(state[:, 1] == SIZE, 1, 0)  # positions full of opponent's pieces

                if player == 0:
                    idx = np.arange(len(state[~done]))
                    _state = from_numpy(augment_data(state[~done]), device)  # (n, 4, 6, 6)
                    _mask = _state[:, 1].reshape(-1, 36) != 0
                    out = cnn[player](_state)
                    _policy = cnn[player].forward_policy_head(out, _mask)  # (n, 36)
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
                    _policy = cnn[player].forward_policy_head(out, _mask)  # (n, 36)
                    policy = _policy.detach().cpu().numpy()

                    action = np.full(env.batch_size, -1)
                    action[~done] = np.concatenate([
                        choice(p) for p in policy
                    ])

                state, done, rewards = env.step(env.state, action)
                player = 1 - player

            if player == 0: rewards = -rewards
            wins += 0.5 * (rewards.sum() + env.batch_size)

        for net in agent.policy_network:
            clip_grad_norm_(net.parameters(), max_norm=1.0, error_if_nonfinite=True)
        policy_optim.step()
        policy_optim.zero_grad()

        win_rate = wins / (2 * env.batch_size)
        print(f'Epoch {ep} | Win Rate: {win_rate * 100:.2f} % | '
              f'elapsed: {time() - start:.2f} s')
        torch.save(agent.state_dict(), 'model/agent')


def rollout(state, action):
    state, done, rewards = env.step(state, action)
    player = 1
    while np.any(~done):
        state[:, [0, 1]] = state[:, [1, 0]]
        state[:, 2] = np.where(state[:, 0] == SIZE, 1, 0)  # positions full of agent's pieces
        state[:, 3] = np.where(state[:, 1] == SIZE, 1, 0)  # positions full of opponent's pieces

        _state = from_numpy(state[~done], device)
        mask = _state[:, 1].reshape(-1, 36) != 0
        out = cnn[player](_state)
        policy = cnn[player].forward_policy_head(out, mask).detach().cpu().numpy()
        action = np.full(env.batch_size, -1)
        action[~done] = np.concatenate([
            choice(p) for p in policy
        ])

        state, done, rewards = env.step(state, action)
        player = 1 - player
    return rewards if player == 1 else -rewards


def play_with_mcts():
    assert env.graphics, 'Interactive mode requires graphics=True.'
    global device
    device = torch.device('cpu')
    for nn in cnn:
        nn.to(device)
        nn.to(device)
    env.batch_size = 1
    env.mode = 'interactive'
    player = np.random.randint(2)
    state, done, rewards = env.reset()
    while np.any(~done):
        state[:, [0, 1]] = state[:, [1, 0]]
        state[:, 2] = np.where(state[:, 0] == SIZE, 1, 0)  # positions full of agent's pieces
        state[:, 3] = np.where(state[:, 1] == SIZE, 1, 0)  # positions full of opponent's pieces

        if player == 0:
            idx = np.arange(len(state[~done]))
            _state = from_numpy(augment_data(state[~done]), device)  # (n, 4, 6, 6)
            _mask = _state[:, 1].reshape(-1, 36) != 0
            out = cnn[player](_state)
            _policy = cnn[player].forward_policy_head(out, _mask)  # (n, 36)
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

            n = from_numpy(augment_data(n.reshape((-1, 1, 6, 6))).reshape((-1, 36)), device)
            target_p = f.softmax(n, dim=1)

            action[~done] = np.concatenate([
                choice(p.cpu().numpy()) for p in target_p[idx]
            ])
        else:
            action = env.wait()

        state, done, rewards = env.step(env.state, action)
        if player == 0:
            env.render(env.state[:4])
        else:
            env.render(env.state[:4, [1, 0]])
        player = 1 - player


if __name__ == '__main__':
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    env = Env(graphics=True, fps=3, batch_size=1024)
    device = torch.device('cuda')
    agent = Network().to(device)
    opponent = Network().to(device)
    cnn = [agent, opponent]

    if os.path.exists('model/agent'):
        model_state = torch.load('model/agent')
        agent.load_state_dict(model_state)

    params = {'weight': [], 'bias': []}
    for net in agent.policy_network:
        for name, param in net.named_parameters():
            if 'bias' in name:
                params['bias'].append(param)
            else:
                params['weight'].append(param)

    policy_optim = torch.optim.NAdam([{'params': params['weight'], 'weight_decay': 1e-3},
                                      {'params': params['bias'], 'weight_decay': 0}],
                                     lr=2e-3)
    value_optim = torch.optim.Adam(agent.value_head.parameters(),
                                   lr=1e-2, weight_decay=1e-4)

    train_with_policy_network()
    # train_with_mcts()
    # play_with_mcts()

    env.close()
