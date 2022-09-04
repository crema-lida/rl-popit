import numpy as np
import torch
# import torch.nn.functional as f
import os
from time import time

from game import Env
from network import Network
from utils import SIZE, from_numpy, choice


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
                state[:, 3] = np.where(state[:, :2].sum(axis=1) == SIZE, 1, 0)  # positions full of pieces

                state = from_numpy(state[~done], device)
                mask = state[:, 1].reshape(-1, 36) != 0
                out = net[player](state)
                policy = net[player].forward_policy_head(out, mask).detach().cpu().numpy()
                action = np.full(env.batch_size, -1)
                action[~done] = np.concatenate([
                    choice(p) for p in policy
                ])

                if player == 0:
                    history['s'].append(state)
                    history['a'].append(action[~done])
                    history['done'].append(done)

                state, done, rewards = env.step(env.state, action)

                if player == 0:
                    if not done[0]: env.paint_canvas(policy[0])
                    env.render(env.state[0])
                else:
                    env.render(env.state[0, [1, 0]])
                player = 1 - player

            if player == 0: rewards = -rewards
            wins += 0.5 * (rewards.sum() + env.batch_size)
            rewards = from_numpy(rewards, device)
            for state, action, done in zip(*history.values()):
                mask = state[:, 1].reshape(-1, 36) != 0
                out = agent(state)
                policy = agent.forward_policy_head(out, mask)
                torch.sum(
                    -torch.log(policy[np.arange(action.size), action]) * rewards[~done]
                ).backward()
                # state_value = agent.forward_value_head(out.detach())
                # value_loss = f.mse_loss(state_value, rewards[~done].unsqueeze(1))
                # value_loss.backward()

        policy_optim.step()
        policy_optim.zero_grad()
        # value_optim.step()
        # value_optim.zero_grad()

        win_rate = wins / (2 * env.batch_size)
        print(f'Epoch {ep} | Win Rate: {win_rate * 100:.2f} % | '
              # f'value loss: {value_loss.item():.4e} | '
              f'elapsed: {time() - start:.2f} s')

        if win_rate > 0.70:
            model_state = agent.state_dict()
            torch.save(model_state, MODEL_PATH)
            net[1].load_state_dict(model_state)
            print('New model state saved!')


def train_with_mcts(epochs=10000):
    start = time()
    for ep in range(epochs + 1):
        wins = 0

        for first_move in range(2):
            player = first_move
            state, done, rewards = env.reset()

            while np.any(~done):
                state[:, [0, 1]] = state[:, [1, 0]]
                state[:, 2] = np.where(state[:, 1] == 0, 1, 0)  # available positions to make a move
                state[:, 3] = np.where(state[:, :2].sum(axis=1) == SIZE, 1, 0)  # positions full of pieces

                state = from_numpy(state[~done], device)
                mask = state[:, 1].reshape(-1, 36) != 0
                out = net[player](state)
                policy = net[player].forward_policy_head(out, mask).detach().cpu().numpy()
                action = np.concatenate([
                    choice(p) for p in policy
                ])

                state, done, rewards = env.step(env.state, action)

                if player == 0:
                    if not done[0]: env.paint_canvas(policy[0])
                    env.render(env.state[0])
                else:
                    env.render(env.state[0, [1, 0]])
                player = 1 - player

            if player == 0: rewards = -rewards
            wins += 0.5 * (rewards.sum() + env.batch_size)

        policy_optim.step()
        policy_optim.zero_grad()

        win_rate = wins / (2 * env.batch_size)
        print(f'Epoch {ep} | Win Rate: {win_rate * 100:.2f} % | '
              # f'value loss: {value_loss.item():.4e} | '
              f'elapsed: {time() - start:.2f} s')

        if win_rate > 0.90:
            model_state = agent.state_dict()
            torch.save(model_state, MODEL_PATH)
            net[1].load_state_dict(model_state)
            print('New model state saved!')


def rollout(state, action):
    player = 0
    state, done, rewards = env.step(state, action)
    while np.any(~done):
        state[:, [0, 1]] = state[:, [1, 0]]
        state[:, 2] = np.where(state[:, 1] == 0, 1, 0)  # available positions to make a move
        state[:, 3] = np.where(state[:, :2].sum(axis=1) == SIZE, 1, 0)  # positions full of pieces

        _state = from_numpy(state[~done], device)
        mask = _state[:, 1].reshape(-1, 36) != 0
        out = net[player](_state)
        policy = net[player].forward_policy_head(out, mask).detach().cpu().numpy()
        action = np.concatenate([
            choice(p) for p in policy
        ])

        state, done, rewards = env.step(state, action)
        player = 1 - player
    return rewards if player == 1 else -rewards


if __name__ == '__main__':
    MODEL_PATH = 'model_v6'

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
                                   lr=0.01, weight_decay=1e-4)
    # train_with_mcts()
    train_with_policy_network()

    env.close()
