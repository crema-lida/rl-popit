import numpy as np
from utils import update_game_state, allocate_spots

BLACK = (61, 60, 66)
WHITE = (255, 255, 255)
GRAY = (207, 210, 207)
COLOR = (84, 186, 185), (255, 91, 0)


class Env:
    def __init__(self, graphics=False, fps: int = None, num_envs=64):
        self.graphics = graphics
        self.fps = fps
        self.num_envs = num_envs
        self.state = np.zeros((self.num_envs, 2, 6, 6), dtype=int)
        self.done = np.full(len(self.state), False)
        self.num_turns = 0
        self.mode = 'train'

        if self.graphics:
            import pygame as pg

            pg.init()
            pg.display.init()
            pg.display.set_caption('Pop it!')
            screen_size = np.array([1440, 1440]) / 2
            self.window = pg.display.set_mode(screen_size)
            self.screen = pg.Surface(screen_size, pg.SRCALPHA)

            self.grid = pg.Surface(screen_size, pg.SRCALPHA)
            pg.draw.line(self.grid, GRAY, (0, 722), (1440, 722), width=5)
            pg.draw.line(self.grid, GRAY, (720, 0), (720, 1440), width=5)
            for i in range(13):
                x = 120 * i
                pg.draw.line(self.grid, GRAY, (0, x), (1440, x), width=3)
                pg.draw.line(self.grid, GRAY, (x, 0), (x, 1440), width=3)

            self.canvas = [pg.Surface((720, 720), pg.SRCALPHA) for _ in range(4)]
            self.rect = [canvas.get_rect() for canvas in self.canvas]
            for i, rect in enumerate(self.rect):
                rect.topleft = (i % 2 * 720, i // 2 * 720)

            self.clock = pg.time.Clock()
            self.font = pg.font.Font(pg.font.get_default_font(), 17)
            self.render()

    def reset(self):
        self.state = np.zeros((self.num_envs, 2, 6, 6), dtype=int)
        self.done = np.full(len(self.state), False)
        self.num_turns = 0
        return self.state.copy(), None, np.full(self.num_envs, False)

    def step(self, state, action, player_idx):
        state = update_game_state(state, action)
        self.state[~self.done] = state if player_idx == 0 else state[:, [1, 0]]
        self.num_turns += 1
        pieces = self.state[:, :2].sum(axis=(2, 3))
        if self.num_turns > 2: self.done = ~pieces.all(axis=1)
        reward = np.where(pieces[:, 0] > 0, 1, -1).astype(np.float32) if np.all(self.done) else None
        return self.state[~self.done].copy(), reward, self.done

    def render(self):
        if not self.graphics: return
        import pygame as pg

        self.screen.fill(WHITE)

        for n, state in enumerate(self.state[:4]):
            canvas, rect = self.canvas[n], self.rect[n]
            self.screen.blit(canvas, rect)
            for player in (0, 1):
                for i, row in enumerate(state[player]):
                    for j, num in enumerate(row):
                        cx, cy = j * 120 + 60 + rect.x, i * 120 + 60 + rect.y
                        spots = allocate_spots(num)
                        for d in spots:
                            pg.draw.circle(self.screen, COLOR[player], (cx + d[0], cy + d[1]), 14)

        self.screen.blit(self.grid, (0, 0))
        size = np.array(self.window.get_rect().size)
        if self.mode == 'interactive':
            size *= 1
        self.window.blit(pg.transform.scale(self.screen, size), (0, 0))
        pg.event.pump()
        pg.display.update()
        if self.fps and self.mode == 'train': self.clock.tick(self.fps)

    def paint_canvas(self, policy):
        if not self.graphics: return
        import pygame as pg

        p = np.zeros((min(self.num_envs, 4), 36))
        sel = ~self.done[:4]
        len_sel = len(np.where(sel)[0])
        p[sel] = policy[:len_sel] / policy[:len_sel].max(axis=1, keepdims=True)

        for j, p in enumerate(p):
            canvas = self.canvas[j]
            for i, alpha in enumerate(p):
                pg.draw.rect(canvas, (220, 214, 247, alpha * 255),
                             pg.Rect(i % 6 * 120, i // 6 * 120, 120, 120))

    def render_text(self, p, q, n):
        if not self.graphics: return

        for j, (p, q, n) in enumerate(zip(p, q, n)):
            canvas = self.canvas[j]
            for i, (_p, _q, _n) in enumerate(zip(p, q, n)):
                texts = f'π: {_p: .4f}', f'Q: {_q: .4f}', f'SEL: {_n: .0f}'
                c = (i % 6 * 120 + 60, i // 6 * 120 + 60)
                position = (c[0], c[1] - 50), (c[0], c[1] - 35), (c[0], c[1] + 50)
                for text, pos in zip(texts, position):
                    text = self.font.render(text, True, BLACK)
                    rect = text.get_rect()
                    rect.center = pos
                    canvas.blit(text, rect)

    @staticmethod
    def wait():
        import pygame as pg
        while True:
            for event in pg.event.get():
                if event.type == pg.MOUSEBUTTONDOWN:
                    action = event.pos[1] // 120 * 6 + event.pos[0] // 120
                    return np.array([action])

    def close(self):
        if not self.graphics: return
        import pygame as pg
        pg.display.quit()
        pg.quit()

    def run(self, model_dir, policy_only=True, policy_weight=1.0, max_searches=180):
        """Run a graphical window to test the model given by model_dir."""

        import onnxruntime
        import torch
        import torch.nn.functional as f
        import time
        import utils

        self.fps = None
        self.mode = 'interactive'
        model = {}
        for module in ['conv_block', 'policy_head', 'value_head']:
            model[module] = onnxruntime.InferenceSession(f'{model_dir}/{module}.onnx')

        player_idx = np.random.randint(2)
        state, reward, done = self.reset()

        while np.any(~done):
            if player_idx == 1:
                state[0, [0, 1]] = state[0, [1, 0]]
                action = self.wait()
            else:
                mask = state[:, 1].reshape(-1, 36) != 0
                out = model['conv_block'].run(None, {
                    'state': utils.zero_center(state).astype(np.float32)
                })[0]
                policy = f.softmax(torch.from_numpy(
                    model['policy_head'].run(None, {
                        'conv.out': out
                    })[0]).masked_fill(torch.from_numpy(mask), -torch.inf), dim=1).numpy()
                if policy_only or self.num_turns < 30:
                    self.paint_canvas(policy)
                    action = np.random.choice(36, size=(1,), p=policy[0])
                else:
                    mask = state[0, 1].reshape(-1, 36) != 0
                    policy[mask] = -np.inf
                    q = np.zeros_like(policy)  # (1, 36) the action value
                    n = np.zeros_like(policy)  # (1, 36)
                    for i in range(max_searches):
                        score = q + policy_weight * policy / (n + 1)  # (1, 36)
                        sel = np.argmax(score, axis=1)
                        q[0, sel] = 0.9 * q[0, sel] + 0.1 * model.rollout(state.copy(), sel, self.num_turns)
                        n[0, sel] += 1
                        exp_cmap = np.exp(score)
                        cmap = exp_cmap / exp_cmap.sum()
                        self.paint_canvas(cmap)
                        self.render_text(policy, q, n)
                        self.render()
                    action = np.argmax(n, axis=-1)

            state, reward, done = self.step(state.copy(), action, player_idx)
            self.render()
            if np.all(done): time.sleep(1)
            player_idx = 1 - player_idx

    def self_play(self, *model_dir, total_games=100):
        """
        Test two models via self-play.
        Given two directories, show the win rate of the first model.
        """

        import onnxruntime
        import torch
        import torch.nn.functional as f
        import utils

        model = [{}, {}]
        for module in ['conv_block', 'policy_head', 'value_head']:
            for i in range(2):
                model[i][module] = onnxruntime.InferenceSession(f'{model_dir[i]}/{module}.onnx')
        wins = 0
        for count in range(total_games):
            player_idx = 0 if count % 2 == 0 else 1
            state, reward, done = self.reset()
            while np.any(~done):
                if player_idx == 1:
                    state[0, [0, 1]] = state[0, [1, 0]]
                mask = state[:, 1].reshape(-1, 36) != 0
                out = model[player_idx]['conv_block'].run(None, {
                    'state': utils.zero_center(state).astype(np.float32)
                })[0]
                policy = f.softmax(torch.from_numpy(
                    model[player_idx]['policy_head'].run(None, {
                        'conv.out': out
                    })[0]).masked_fill(torch.from_numpy(mask), -torch.inf), dim=1).numpy()
                action = np.random.choice(36, size=(1,), p=policy[0])

                state, reward, done = self.step(state.copy(), action, player_idx)
                player_idx = 1 - player_idx
            if reward[0] == 1:
                wins += 1
            print(f'\r{count + 1}/{total_games} | Model 1 Win Rate: {wins / (count + 1) * 100:.2f}%', end='')
