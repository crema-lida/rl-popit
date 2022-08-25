import numpy as np
from numba import njit, prange

SIZE = np.full((6, 6), 3, dtype=int)
SIZE[::5, :] = 2
SIZE[:, ::5] = 2
SIZE[::5, ::5] = 1


@njit
def choice(p):
    return np.searchsorted(np.cumsum(p), np.random.rand(1))


@njit
def update(state, player, pos):
    state[player][pos] += 1

    if state[player][pos] > SIZE[pos]:
        state[player][pos] = 0
        neighbors = []
        i, j = pos
        for i, j in ((i, j + 1), (i + 1, j), (i, j - 1), (i - 1, j)):
            if 0 <= i < 6 and 0 <= j < 6:
                neighbors.append((i, j))

        for pos in neighbors:
            state[player][pos] += state[1 - player][pos]
            state[1 - player][pos] = 0
            if state[1 - player].sum() == 0: return
            update(state, player, pos)


@njit(parallel=True)
def update_batch(state, player, action):
    for i in prange(len(action)):
        if action[i] == -1: continue
        update(state[i], player, (action[i] // 6, action[i] % 6))


def allocate_spots(num):
    if num == 1:
        spots = (0, 0),
    elif num == 2:
        spots = (-17, 0), (17, 0)
    elif num == 3:
        spots = (0, -15), (-17, 15), (17, 15)
    else:
        spots = ()
    return spots
