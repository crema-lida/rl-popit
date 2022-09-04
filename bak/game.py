import numpy as np
from utils import SIZE, from_numpy, update_batch, allocate_spots

WHITE = (255, 255, 255)
GREEN = (206, 229, 208)
COLOR = (84, 186, 185), (255, 91, 0)


class Player:
    def __init__(self, net, batch_size, device, R: int):
        self.net = net.to(device)
        self.device = device
        self.R = R
        self.rewards = np.zeros((batch_size, 0))
        self.state = np.zeros((batch_size, 4, 6, 6), dtype=int)
        self.policy = None
        self.action = None

    def observe_state(self):
        self.state[:, [0, 1]] = self.state[:, [1, 0]]
        self.state[:, 2] = np.where(self.state[:, 1] == 0, 1, 0)  # available positions to make a move
        self.state[:, 3] = np.where(self.state[:, :2].sum(axis=1) == SIZE, 1, 0)  # positions full of pieces

    def make_policy(self, state=None):
        if state is None:
            state = self.state
        if len(state) == 0: return
        state = from_numpy(state, self.device)
        out = self.net(state)
        mask = state[:, 1].reshape(-1, 36) != 0
        self.policy = self.net.forward_policy_head(out, mask)


class Env:
    def __init__(self, graphics=True, fps: int = None, batch_size=256):
        self.graphics = graphics
        self.fps = fps
        self.batch_size = batch_size
        self.sizes = np.full((6, 6), 4, dtype=int)
        self.turn = 0
        self.win_rate = 0.0

        if self.graphics:
            import pygame as pg

            pg.init()
            pg.display.init()
            pg.display.set_caption('Pop it!')
            self.window = pg.display.set_mode((720, 720))
            self.canvas = pg.Surface((720, 720), pg.SRCALPHA)
            self.clock = pg.time.Clock()

    def reset(self):
        info = f'{self.pieces[0]} : {self.pieces[1]} | Win Rate: {self.win_rate * 100:.2f} %'
        self.turn = 0
        return info

    def step(self, agent, opp):
        for player in (agent, opp):
            if len(player.state) == 0: continue
            player.state = update_batch(player.state, player.action)  # calculate new state caused by player's action
            pieces = player.state[:, :2].sum(axis=(2, 3))
            rewards = np.zeros((self.batch_size, 1))
            if pieces[0].sum() > 2:
                done = ~pieces.all(axis=1)
                if np.any(done):
                    choice_unfinished = np.all(player.rewards == 0, axis=1)
                    unfinished = rewards[choice_unfinished]
                    unfinished[done] = player.R
                    rewards[choice_unfinished] = unfinished
                    player.state = player.state[~done]  # delete finished states
            player.rewards = np.concatenate((player.rewards, rewards), axis=1)

        if self.graphics:
            if self.turn == 0:
                self.render(agent.state[0])
            else:
                self.render(opp.state[0, [1, 0, 2, 3]])

        self.turn = 1 - self.turn
        agent.state, opp.state = opp.state, agent.state
        agent.rewards, opp.rewards = opp.rewards, agent.rewards
        if len(agent.state) + len(opp.state) == 0:
            done = True
        else:
            done = False
        return done

    def render(self, state):
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
        if hasattr(self, 'window'):
            import pygame as pg

            pg.display.quit()
            pg.quit()
