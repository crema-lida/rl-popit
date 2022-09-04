if __name__ == '__main__':
    import numpy as np
    import torch
    import os
    from time import time

    from game import Env, Player
    from network import Network
    from utils import from_numpy, choice

    MODEL_PATH = 'model_v6'

    env = Env(graphics=True, fps=None, batch_size=256)
    device = torch.device('cuda')
    agent = Player(Network(), env.batch_size // 2, device, 1)
    opp = Player(Network(), env.batch_size // 2, device, -1)
    if os.path.exists(MODEL_PATH):
        model_state = torch.load(MODEL_PATH)
        print(f'Model state loaded from {MODEL_PATH}')
        for net in (agent.net, opp.net):
            net.load_state_dict(model_state)
    policy_optim = torch.optim.Adam([{'params': agent.net.conv_block.parameters()},
                                     {'params': agent.net.res_net.parameters()},
                                     {'params': agent.net.policy_head.parameters()}],
                                    lr=0.01, weight_decay=1e-4)
    value_optim = torch.optim.Adam(agent.net.value_head.parameters(),
                                   lr=0.01, weight_decay=1e-4)
    epoch = 1
    # keeps a record of states and actions
    history = {'s': [], 'a': [], 'idx': [],  'done': []}

    start = time()
    while epoch <= 10000:
        for player in (agent, opp):
            player.observe_state()
            player.make_policy()
            player.action = np.concatenate([choice(p) for p in player.policy.detach().cpu().numpy()])

        if len(agent.state) != 0:
            history['s'].append(agent.state)
            history['a'].append(agent.action)
            history['idx'].append(agent.idx)
        if env.turn == 0:
            env.paint_canvas(agent.policy[0])

        done = env.step(agent, opp)

        if done:
            rewards = from_numpy(rewards, device)
            for a, (state, action, idx, done) in enumerate(zip(*history.values())):
                agent.make_policy(state)
                torch.sum(
                    -torch.log(agent.policy[np.arange(action.size), action]) * rewards[idx][~done]  # * 1.1 ** a
                ).backward()
                # state_value = agent.forward_value_head(out.detach())
                # value_loss = f.mse_loss(state_value, rewards[~done].unsqueeze(1))
                # value_loss.backward()
            policy_optim.step()
            policy_optim.zero_grad()
            # value_optim.step()
            # value_optim.zero_grad()
            win_rate = 0.5 * (rewards.sum() + self.batch_size) / self.batch_size
            info = env.reset()
            agent.idx = slice(env.batch_size // 2)
            opp.idx = slice(env.batch_size // 2, env.batch_size)
            for arr in history.values():
                arr.clear()
            print(f'Epoch {epoch} |', info,
                  # f'| value loss: {value_loss.item():.4e}'
                  f'| elapsed: {time() - start:.2f} s')
            epoch += 1

            if epoch % 20 == 0:
                if env.win_rate > 0.70:
                    model_state = agent.net.state_dict()
                    torch.save(model_state, MODEL_PATH)
                    opp.net.load_state_dict(model_state)
                    print('New model state saved!')

    env.close()
