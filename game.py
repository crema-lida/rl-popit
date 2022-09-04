import numpy as np
from utils import update_batch, allocate_spots

WHITE = (255, 255, 255)
GREEN = (206, 229, 208)
COLOR = (84, 186, 185), (255, 91, 0)


class Env:
    def __init__(self, graphics=False, fps=30, batch_size=2):
        self.graphics = graphics
        self.fps = fps
        self.batch_size = batch_size
        self.state = np.zeros((self.batch_size, 4, 6, 6), dtype=int)
        self.sizes = np.full((6, 6), 4, dtype=int)

        if self.graphics:
            import pygame as pg

            pg.init()
            pg.display.init()
            pg.display.set_caption('Pop it!')
            self.window = pg.display.set_mode((720, 720))
            self.canvas = pg.Surface((720, 720), pg.SRCALPHA)
            self.clock = pg.time.Clock()

    def reset(self):
        self.state.fill(0)
        return self.state, np.full(self.batch_size, False), None

    def step(self, state, action):
        update_batch(state, action)

        pieces = state[:, :2].sum(axis=(2, 3))
        done = ~pieces.all(axis=1) if pieces[0].sum() > 2 else np.full(self.batch_size, False)
        rewards = np.where(pieces[:, 0] > 0, 1, -1) if np.all(done) else None
        return state, done, rewards

    def render(self, state):
        if not self.graphics: return
        import pygame as pg

        self.window.fill(WHITE)
        self.window.blit(self.canvas, (0, 0))

        for i in range(7):
            x = 120 * i
            pg.draw.line(self.window, GREEN, (0, x), (720, x), width=3)
            pg.draw.line(self.window, GREEN, (x, 0), (x, 720), width=3)

        for player in (0, 1):
            for i, row in enumerate(state[player]):
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
        import pygame as pg

        for i, alpha in enumerate(p):
            pg.draw.rect(self.canvas, (149, 205, 65, alpha * 255),
                         pg.Rect(i % 6 * 120, i // 6 * 120, 120, 120))

    def close(self):
        if not self.graphics: return
        import pygame as pg
        pg.display.quit()
        pg.quit()
