if __name__ == '__main__':
    import numpy as np
    import torch
    import torch.nn.functional as f
    import os
    from time import time

    from game import Env
    from network import Network
    from utils import SIZE, choice

    MODEL_NAME = 'model_v5'

    env = Env(graphics=True, fps=None, batch_size=512)
    state = env.state
    device = torch.device('cuda')
    features = state.shape[1]
    agent = Network(features).to(device)
    opponent = Network(features).to(device)
    net = [agent, opponent]
    if os.path.exists(MODEL_NAME):
        print(MODEL_NAME, 'loaded.')
        model_state = torch.load(MODEL_NAME)
        agent.load_state_dict(model_state)
    opponent.load_state_dict(agent.state_dict())
    policy_optim = torch.optim.Adam([{'params': agent.conv_block.parameters()},
                                     {'params': agent.res_net.parameters()},
                                     {'params': agent.policy_head.parameters()}],
                                    lr=0.001, weight_decay=1e-4)
    value_optim = torch.optim.Adam(agent.value_head.parameters(),
                                   lr=0.001, weight_decay=1e-4)
    player = np.random.choice(2)  # idx=0 for agent and idx=1 for opponent
    epoch = 1
    # keeps a record of states and actions
    history = {'s': [], 'a': [], 'done': []}

    start = time()
    while epoch <= 10000:
        if player == 1:
            state = state.copy()
            state[:, [0, 1]] = state[:, [1, 0]]
        state[:, 2] = np.where(state[:, 1] == 0, 1, 0)  # available positions to make a move
        state[:, 3] = np.where(state[:, :2].sum(axis=1) == SIZE, 1, 0)
        state = torch.from_numpy(state[~env.done]).to(device=device, dtype=torch.float, non_blocking=True)
        if player == 0:
            history['done'].append(env.done)
            history['s'].append(state)
        mask = state[:, 1].reshape(-1, 36) != 0
        out = net[player](state)
        policy = net[player].forward_policy_head(out, mask)
        _policy = policy.detach().cpu().numpy()
        action = np.concatenate([
            choice(p) for p in _policy
        ])

        if player == 0:
            if not env.done[0]: env.paint_canvas(_policy[0])
            history['a'].append(action)

        state, rewards, info = env.step(player, action)

        player = 1 - player  # switch to the other player
        if all(env.done):
            rewards = torch.from_numpy(rewards).to(device=device, dtype=torch.float, non_blocking=True)
            for state, action, done in zip(*history.values()):
                mask = state[:, 1].reshape(-1, 36) != 0
                out = agent(state)
                policy = agent.forward_policy_head(out, mask)
                sum(-torch.log(policy[np.arange(action.size), action]) * rewards[~done]).backward()
                # state_value = agent.forward_value_head(out.detach())
                # value_loss = f.mse_loss(state_value, rewards[~done].unsqueeze(1))
                # value_loss.backward()
            policy_optim.step()
            policy_optim.zero_grad()
            # value_optim.step()
            # value_optim.zero_grad()
            state, done_info = env.reset()
            for arr in history.values():
                arr.clear()
            player = np.random.choice(2)
            print(f'Epoch {epoch} |', info + done_info,
                  # f'| value loss: {value_loss.item():.4e}'
                  f'| elapsed: {time() - start:.2f} s')
            epoch += 1

            if epoch % 20 == 0:
                if env.win_rate > 0.70:
                    model_state = agent.state_dict()
                    torch.save(model_state, MODEL_NAME)
                    net[1].load_state_dict(model_state)
                    print('New model state saved!')
                    env.total_games = 0
                    env.wins = 0

    env.close()
