import numpy as np
import torch
from numba import njit, prange


SIZE = np.full((6, 6), 3, dtype=int)
SIZE[::5, :] = 2
SIZE[:, ::5] = 2
SIZE[::5, ::5] = 1


def from_numpy(arr, device=torch.device('cpu')):
    return torch.from_numpy(arr).to(device=device, dtype=torch.float, non_blocking=True)


@njit
def choice(p):
    return np.searchsorted(np.cumsum(p), np.random.rand(1))  # select one integer from range(len(p)) according to p


@njit
def update(state, pos):
    state[0][pos] += 1

    if state[0][pos] > SIZE[pos]:
        state[0][pos] = 0
        neighbors = []
        i, j = pos
        for i, j in ((i, j + 1), (i + 1, j), (i, j - 1), (i - 1, j)):
            if 0 <= i < 6 and 0 <= j < 6:
                neighbors.append((i, j))

        for pos in neighbors:
            state[0][pos] += state[1][pos]
            state[1][pos] = 0
            if state[1].sum() == 0: return
            update(state, pos)


@njit
def update_batch(state, action):
    for i in prange(len(state)):
        update(state[i], (action[i] // 6, action[i] % 6))
    return state


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
