import numpy as np
import torch
import os, time
from tqdm.autonotebook import tqdm

from globals import glob


def train_with_ppo(epochs=20000, sample_loops=10, drop_threshold=85):
    writer, env, agent, opponent = glob.conf
    cnn = (agent.model, opponent)
    ep_start, ema_win_rate = agent.load_checkpoint()

    with tqdm(total=epochs, ncols=80, position=0, desc='Progress') as tbar_main:
        tbar_main.update(ep_start)
        for ep in range(ep_start, epochs):
            tbar_main.set_postfix_str(f'lr={agent.scheduler.get_last_lr()[0]:.2e}')
            tbar_main.update(1)
            if (ep <= 1000 and ep % 100 == 0 or ep > 1000 and ep % 500 == 0) and \
                    np.all(np.array(list(ema_win_rate.values())) > 55):
                agent.save_as_opponent(glob.MODEL_DIR, ep)

            # sample (loops * batch_size) games by playing with older models
            agent.model.eval()
            with tqdm(total=sample_loops, ncols=80, position=1, leave=False, desc='Self Play') as tbar:
                for _ in range(sample_loops // 2):
                    dirlist = os.listdir(f'{glob.MODEL_DIR}/opp')
                    model_name = np.random.choice(dirlist)  # randomly select a model from history
                    opponent.load_state_dict(torch.load(f'{glob.MODEL_DIR}/opp/{model_name}'))
                    wins = 0
                    for first_move in range(2):
                        tbar.update(1)
                        player_idx = first_move
                        state, reward, done = env.reset()
                        while np.any(~done):
                            if player_idx == 1:
                                state[:, [0, 1]] = state[:, [1, 0]]
                            policy, action = agent.choose_action(cnn[player_idx], state)
                            state_new, reward, done_new = env.step(state.copy(), action, player_idx)

                            if player_idx == 0:
                                agent.remember(state, policy, action, done)
                            elif env.graphics:
                                policy, _ = agent.choose_action(agent.model, state_new)
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
                    if ema_win_rate[model_name] > drop_threshold:
                        # remove the old model, and save the latest for opponent to use
                        os.remove(f'{glob.MODEL_DIR}/opp/{model_name}')
                        agent.save_as_opponent(glob.MODEL_DIR, ep)
                        del ema_win_rate[model_name]

            agent.model.train()
            info = agent.learn()
            agent.scheduler.step()
            for tag, data in info.items():
                if tag == 'value_loss':
                    writer.add_scalar(tag, torch.concat(data).mean(), ep)
                else:
                    writer.add_histogram(f'histogram/{tag}', torch.concat(data), ep)
            writer.add_scalars('win_rate', ema_win_rate, ep)

            log = '\n' + f'Epoch {ep}'.center(45) + '\n' + 'Opponent      EMA Win Rate (%)'.center(40)
            for model_name, win_rate in ema_win_rate.items():
                log += '\n' + f'{model_name}     {win_rate: .2f}'.center(40)
            tqdm.write(log)

            if ep % 10 == 0:
                agent.save_checkpoint(glob.MODEL_DIR, ep, ema_win_rate)


def train_with_mcts(): ...


if __name__ == '__main__':
    from torch.utils.tensorboard import SummaryWriter
    import network
    from game import Env
    from agent import Agent

    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    glob.MODEL_DIR = 'cnn2-64'
    glob.SUMMARY_DIR = 'runs/' + time.asctime().replace(' ', '-').replace(':', '.')
    writer = SummaryWriter(glob.SUMMARY_DIR)
    glob.device = torch.device('cuda')

    env = Env(num_envs=128)
    agent = Agent(network.CNN(features=64, num_blocks=2).to(glob.device),
                  minibatch_size=2048, clip=0.1, entropy_coeff=0.01, max_norm=0.5,
                  lr=1e-3, weight_decay=1e-2, eps=1e-5)
    opponent = network.CNN(features=64, num_blocks=2).to(glob.device).eval()
    glob.conf = (writer, env, agent, opponent)

    train_with_ppo(epochs=15000, sample_loops=20, drop_threshold=90)

    env.close()
