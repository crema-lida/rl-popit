import numpy as np
import torch
from numba import njit, guvectorize, int64

SIZE = np.full((6, 6), 3, dtype=int)
SIZE[::5, :] = 2
SIZE[:, ::5] = 2
SIZE[::5, ::5] = 1

device = torch.device('cpu')


def to_mask(indices):
    """convert indices (n,) to masks (n, 36)"""
    size = len(indices)
    mask = np.full((size, 36), False)
    mask[np.arange(size), indices] = True
    return mask


def zero_center(arr):
    return arr - np.mean(arr, axis=(2, 3), keepdims=True)


@njit
def augment_data(arr):
    flipud, fliplr = arr[..., ::-1, :], arr[..., ::-1]
    rot_180 = flipud[..., ::-1]
    rot_90l = np.transpose(arr[..., ::-1], (0, 1, 3, 2))
    rot_90r = np.transpose(arr[..., ::-1, :], (0, 1, 3, 2))
    return np.concatenate((arr, flipud, fliplr, rot_180, rot_90l, rot_90r))


@guvectorize([(int64[:, :, :], int64, int64[:, :, :])],
             '(C,K,K),()->(C,K,K)',
             target='parallel', nopython=True)
def update_game_state(state, action, new_state):
    new_state[:] = state[:]
    indices = {action: 1}
    while len(indices) > 0:
        idx, n = indices.popitem()
        i, j = idx // 6, idx % 6
        new_state[0][i, j] += n

        if new_state[0][i, j] > SIZE[i, j]:
            new_state[0][i, j] -= SIZE[i, j] + 1
            neighbors = []
            for i, j in ((i, j + 1), (i + 1, j), (i, j - 1), (i - 1, j)):
                if 0 <= i < 6 and 0 <= j < 6:
                    neighbors.append((i, j))
            for i, j in neighbors:
                new_state[0][i, j] += new_state[1][i, j]
                new_state[1][i, j] = 0
                if new_state[1].sum() == 0: return
                idx = i * 6 + j
                indices[idx] = 1 if idx not in indices else indices[idx] + 1


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
