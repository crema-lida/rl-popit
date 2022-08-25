import numpy as np
import pygame as pg
from utils import update_batch, allocate_spots

WHITE = (255, 255, 255)
GREEN = (206, 229, 208)
COLOR = (84, 186, 185), (255, 91, 0)


class Env:
    def __init__(self, graphics=False, fps=30, batch_size=1):
        self.graphics = graphics
        self.fps = fps
        self.batch_size = batch_size
        self.state = np.zeros((batch_size, 4, 6, 6), dtype=int)
        self.sizes = np.full((6, 6), 4, dtype=int)
        self.pieces = self.state.sum(axis=(2, 3))
        self.done = np.full(batch_size, False)
        self.round = 0  # keeps a record of number of plays in one game
        self.total_games = 0
        self.wins = 0
        self.win_rate = 0.0

        if self.graphics:
            pg.init()
            pg.display.init()
            pg.display.set_caption('Pop it!')
            self.window = pg.display.set_mode((720, 720))
            self.canvas = pg.Surface((720, 720), pg.SRCALPHA)
            self.clock = pg.time.Clock()

    def reset(self):
        self.win_rate = self.wins / self.total_games
        info = f'Win Rate: {self.win_rate * 100:.2f} %'
        self.state.fill(0)
        self.round = 0
        self.pieces.fill(0)
        self.done = np.full(self.batch_size, False)
        return self.state, info

    def step(self, player: int, action: np.ndarray):
        _action = np.full(self.batch_size, -1)
        _action[~self.done] = action
        update_batch(self.state, player, _action)

        pieces = self.state[:, :2].sum(axis=(2, 3))
        rewards = 0
        self.round += 0.5
        if self.round > 1:
            self.done = ~pieces.all(axis=1)
        if all(self.done):
            rewards = np.where(pieces[:, 0] > 0, 1, -1)
            self.total_games += self.batch_size
            self.wins += 0.5 * (rewards.sum() + self.batch_size)
        info = f'{pieces[0][0]} : {pieces[0][1]} | '
        self.pieces = pieces

        if self.graphics: self.render()
        return self.state, rewards, info

    def render(self):
        self.window.fill(WHITE)
        self.window.blit(self.canvas, (0, 0))

        for i in range(7):
            x = 120 * i
            pg.draw.line(self.window, GREEN, (0, x), (720, x), width=3)
            pg.draw.line(self.window, GREEN, (x, 0), (x, 720), width=3)

        for player in (0, 1):
            for i, row in enumerate(self.state[0][player]):
                for j, num in enumerate(row):
                    cx, cy = j * 120 + 60, i * 120 + 60
                    spots = allocate_spots(num)
                    for d in spots:
                        pg.draw.circle(self.window, COLOR[player], (cx + d[0], cy + d[1]), 14)

        pg.event.pump()
        pg.display.update()
        if self.fps: self.clock.tick(self.fps)

    def paint_canvas(self, p):
        if not self.graphics: return
        for i, alpha in enumerate(p):
            pg.draw.rect(self.canvas, (149, 205, 65, alpha * 255),
                         pg.Rect(i % 6 * 120, i // 6 * 120, 120, 120))

    def close(self):
        if hasattr(self, 'window'):
            pg.display.quit()
            pg.quit()
