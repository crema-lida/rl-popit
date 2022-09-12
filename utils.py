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
    """select one integer from range(len(p)) according to p"""
    return np.searchsorted(np.cumsum(p), np.random.rand(1)).clip(None, 35)


def augment_data(arr):
    flipud, fliplr = np.flip(arr, 2), np.flip(arr, 3)
    rot_180 = np.flip(arr, (2, 3))
    rot_90l = np.transpose(arr[..., ::-1], (0, 1, 3, 2))
    rot_90r = np.flip(rot_90l, (2, 3))
    return np.concatenate((arr, flipud, fliplr, rot_180, rot_90l, rot_90r))


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


@njit(parallel=True)
def update_batch(state, action):
    for i in prange(len(state)):
        if action[i] == -1: continue
        update(state[i], (action[i] // 6, action[i] % 6))


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
