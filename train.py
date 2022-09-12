import numpy as np
import torch
# import torch.nn.functional as f
import os
from time import time

from game import Env
from network import Network
from utils import SIZE, from_numpy, choice, augment_data


def train_with_policy_network(epochs=10000):
    start = time()
    for ep in range(epochs + 1):
        wins = 0

        # Agent makes the first move in the first half batch,
        # while opponent makes the first move in the second half.
        for first_move in range(2):
            player = first_move
            history = {'s': [], 'a': [], 'done': []}  # keeps a record of states and actions
            state, done, rewards = env.reset()

            while np.any(~done):
                state[:, [0, 1]] = state[:, [1, 0]]
                state[:, 2] = np.where(state[:, 1] == 0, 1, 0)  # available positions to make a move
                state[:, 3] = np.where(state[:, 0] == SIZE, 1, 0)  # positions full of agent's pieces
                state[:, 4] = np.where(state[:, 1] == SIZE, 1, 0)  # positions full of opponent's pieces

                state = from_numpy(state[~done], device)  # (n, 4, 6, 6)
                mask = state[:, 1].reshape(-1, 36) != 0
                out = net[player](state)
                policy = net[player].forward_policy_head(out, mask).detach().cpu().numpy()  # (n, 36)
                action = np.full(env.batch_size, -1)  # (N,)
                action[~done] = np.concatenate([
                    choice(p) for p in policy
                ])

                if player == 0:
                    history['s'].append(env.state[~done])
                    history['a'].append(action[~done])
                    history['done'].append(done)

                state, done, rewards = env.step(env.state, action)

                if player == 0:
                    p = policy[:4] / policy[:4].max(axis=1, keepdims=True)
                    env.paint_canvas(p)
                    env.render(env.state[:4])
                else:
                    env.render(env.state[:4, [1, 0]])
                player = 1 - player

            if player == 0: rewards = -rewards
            wins += 0.5 * (rewards.sum() + env.batch_size)
            rewards = from_numpy(rewards, device)
            for state, action, done in zip(*history.values()):
                size = len(state)
                act = np.full((size, 36), False)
                act[np.arange(size), action] = True

                action = torch.from_numpy(augment_data(act.reshape((-1, 1, 6, 6))).reshape((-1, 36)))
                state = from_numpy(augment_data(state), device)
                mask = state[:, 1].reshape(-1, 36) != 0
                out = agent(state)
                policy = agent.forward_policy_head(out, mask)
                torch.sum(
                    -torch.log(policy[action]) * rewards[~done].repeat(6)
                ).backward()
                # for param in agent.conv_block.parameters():
                #     print(param.grad)
                # state_value = agent.forward_value_head(out.detach())
                # value_loss = f.mse_loss(state_value, rewards[~done].unsqueeze(1))
                # value_loss.backward()

        policy_optim.step()
        policy_optim.zero_grad()
        # value_optim.step()
        # value_optim.zero_grad()

        win_rate = wins / (2 * env.batch_size)
        print(f'Epoch {ep} | Win Rate: {win_rate * 100:.2f} % | '
              f'elapsed: {time() - start:.2f} s')

        if win_rate > 0.65:
            model_state = agent.state_dict()
            torch.save(model_state, MODEL_PATH)
            net[1].load_state_dict(model_state)
            print('New model state saved!')


def train_with_mcts(epochs=10000):
    cross_entropy = torch.nn.CrossEntropyLoss()
    start = time()
    for ep in range(epochs + 1):
        wins = 0

        for first_move in range(2):
            player = first_move
            state, done, rewards = env.reset()

            while np.any(~done):
                state[:, [0, 1]] = state[:, [1, 0]]
                state[:, 2] = np.where(state[:, 1] == 0, 1, 0)
                state[:, 3] = np.where(state[:, 0] == SIZE, 1, 0)  # positions full of agent's pieces
                state[:, 4] = np.where(state[:, 1] == SIZE, 1, 0)  # positions full of opponent's pieces

                if player == 0:
                    idx = np.arange(len(state[~done]))
                    _state = from_numpy(augment_data(state[~done]), device)  # (n, 4, 6, 6)
                    mask = _state[:, 1].reshape(-1, 36) != 0
                    out = net[player](_state)
                    _policy = net[player].forward_policy_head(out, mask)  # (n, 36)
                    policy = _policy[idx].detach().cpu().numpy()

                    action_value = np.zeros_like(policy)  # (n, 36)
                    n = np.zeros_like(policy)  # (n, 36)
                    action = np.full(env.batch_size, -1)  # (N,)

                    for i in range(500):
                        q = action_value.clip(min=0) / (1e-5 + n)  # (n, 36)
                        score = q + 0.01 * policy / (n + 1)  # (n, 36)
                        selection = np.argmax(score, axis=1)
                        action[~done] = selection
                        action_value[idx, selection] += rollout(state.copy(), action)[~done]
                        n[idx, selection] += 1

                        p = score[:4] / np.sum(score[:4], axis=1, keepdims=True)
                        env.paint_canvas(p / p.max(axis=1, keepdims=True))
                        env.render_text(policy[:4], q[:4], n[:4])
                        env.render(env.state[:4])

                    n = from_numpy(augment_data(n.reshape((-1, 1, 6, 6))).reshape((-1, 36)), device)
                    target_p = n / torch.sum(n, dim=1, keepdim=True)
                    loss = cross_entropy(_policy, target_p)
                    loss.backward()
                    policy_optim.step()
                    policy_optim.zero_grad()

                    net[1].load_state_dict(agent.state_dict())

                    action[~done] = np.concatenate([
                        choice(p.cpu().numpy()) for p in target_p[idx]
                    ])
                else:
                    _state = from_numpy(state[~done], device)
                    mask = _state[:, 1].reshape(-1, 36) != 0
                    out = net[player](_state)
                    _policy = net[player].forward_policy_head(out, mask)  # (n, 36)
                    policy = _policy.detach().cpu().numpy()

                    action = np.full(env.batch_size, -1)
                    action[~done] = np.concatenate([
                        choice(p) for p in policy
                    ])

                state, done, rewards = env.step(env.state, action)
                player = 1 - player

            if player == 0: rewards = -rewards
            wins += 0.5 * (rewards.sum() + env.batch_size)

        win_rate = wins / (2 * env.batch_size)
        print(f'Epoch {ep} | Win Rate: {win_rate * 100:.2f} % | '
              f'elapsed: {time() - start:.2f} s')

        model_state = agent.state_dict()
        torch.save(model_state, MODEL_PATH)
        print('New model state saved!')


def rollout(state, action):
    state, done, rewards = env.step(state, action)
    player = 1
    while np.any(~done):
        state[:, [0, 1]] = state[:, [1, 0]]
        state[:, 2] = np.where(state[:, 1] == 0, 1, 0)  # available positions to make a move
        state[:, 3] = np.where(state[:, 0] == SIZE, 1, 0)  # positions full of agent's pieces
        state[:, 4] = np.where(state[:, 1] == SIZE, 1, 0)  # positions full of opponent's pieces

        _state = from_numpy(state[~done], device)
        mask = _state[:, 1].reshape(-1, 36) != 0
        out = net[player](_state)
        policy = net[player].forward_policy_head(out, mask).detach().cpu().numpy()
        action = np.full(env.batch_size, -1)
        action[~done] = np.concatenate([
            choice(p) for p in policy
        ])

        state, done, rewards = env.step(state, action)
        player = 1 - player
    return rewards if player == 1 else -rewards


if __name__ == '__main__':
    MODEL_PATH = 'model_v8'
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    env = Env(graphics=False, fps=None, batch_size=512)
    device = torch.device('cuda')
    agent = Network().to(device)
    opponent = Network().to(device)
    net = [agent, opponent]
    if os.path.exists(MODEL_PATH):
        model_state = torch.load(MODEL_PATH)
        print(f'Model state loaded from {MODEL_PATH}')
        agent.load_state_dict(model_state)
    opponent.load_state_dict(agent.state_dict())
    policy_optim = torch.optim.Adam([{'params': agent.conv_block.parameters()},
                                     {'params': agent.res_net.parameters()},
                                     {'params': agent.policy_head.parameters()}],
                                    lr=0.01, weight_decay=1e-4)
    value_optim = torch.optim.Adam(agent.value_head.parameters(),
                                   lr=0.001, weight_decay=1e-4)

    train_with_policy_network()
    # train_with_mcts()

    env.close()
