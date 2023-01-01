import onnxruntime
import numpy as np
import torch
import torch.nn.functional as f
import time

import utils


def run(model_dir, policy_only=True, policy_weight=1.0, max_searches=180):
    env = Env(graphics=True, fps=None, num_envs=1)
    env.mode = 'interactive'
    model = {}
    for module in ['conv_block', 'policy_head', 'value_head']:
        model[module] = onnxruntime.InferenceSession(f'{model_dir}/{module}.onnx')

    player_idx = np.random.randint(2)
    state, reward, done = env.reset()

    while np.any(~done):
        if player_idx == 1:
            state[0, [0, 1]] = state[0, [1, 0]]
            action = env.wait()
        else:
            mask = state[:, 1].reshape(-1, 36) != 0
            out = model['conv_block'].run(None, {'state': utils.zero_center(state).astype(np.float32)})[0]
            policy = f.softmax(
                torch.from_numpy(
                    model['policy_head'].run(None, {'conv.out': out})[0]
                ).masked_fill(torch.from_numpy(mask), -torch.inf),
                dim=1,
            ).numpy()
            if policy_only or env.num_turns < 30:
                env.paint_canvas(policy)
                action = np.random.choice(36, size=(1,), p=policy[0])
            else:
                mask = state[0, 1].reshape(-1, 36) != 0
                policy[mask] = -np.inf
                q = np.zeros_like(policy)  # (1, 36) the action value
                n = np.zeros_like(policy)  # (1, 36)
                for i in range(max_searches):
                    score = q + policy_weight * policy / (n + 1)  # (1, 36)
                    sel = np.argmax(score, axis=1)
                    q[0, sel] = 0.9 * q[0, sel] + 0.1 * model.rollout(state.copy(), sel, env.num_turns)
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


def self_play(*model_dir, total_games=100):
    env = Env(num_envs=1)
    model = [{}, {}]
    for module in ['conv_block', 'policy_head', 'value_head']:
        for i in range(2):
            model[i][module] = onnxruntime.InferenceSession(f'{model_dir[i]}/{module}.onnx')
    wins = 0
    for count in range(total_games):
        player_idx = 0 if count % 2 == 0 else 1
        state, reward, done = env.reset()
        while np.any(~done):
            if player_idx == 1:
                state[0, [0, 1]] = state[0, [1, 0]]
            mask = state[:, 1].reshape(-1, 36) != 0
            out = model[player_idx]['conv_block'].run(None, {'state': utils.zero_center(state).astype(np.float32)})[0]
            policy = f.softmax(
                torch.from_numpy(
                    model[player_idx]['policy_head'].run(None, {'conv.out': out})[0]
                ).masked_fill(torch.from_numpy(mask), -torch.inf),
                dim=1,
            ).numpy()
            action = np.random.choice(36, size=(1,), p=policy[0])

            state, reward, done = env.step(state.copy(), action, player_idx)
            player_idx = 1 - player_idx
        if reward[0] == 1:
            wins += 1
        print(f'\r{count + 1}/{total_games} | Model 1 Win Rate: {wins / (count + 1) * 100:.2f}%', end='')


if __name__ == '__main__':
    from game import Env

    run('../best_models/cnn2-64')
    # self_play('../best_models/cnn2-64', '../best_models/resnet3-64')
