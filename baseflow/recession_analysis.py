import numpy as np
from numba import njit
from utils import moving_average, multi_arange


@njit
def recession_period(Q):
    idx_dec = np.zeros(Q.shape[0] - 1, dtype=np.int64)
    Q_ave = moving_average(Q, 3)
    idx_dec[1:-1] = (Q_ave[:-1] - Q_ave[1:]) > 0
    idx_beg = np.where(idx_dec[:-1] - idx_dec[1:] == -1)[0] + 1
    idx_end = np.where(idx_dec[:-1] - idx_dec[1:] == 1)[0] + 1
    idx_keep = (idx_end - idx_beg) >= 10
    idx_beg = idx_beg[idx_keep]
    idx_end = idx_end[idx_keep]
    duration = idx_end - idx_beg
    idx_beg = idx_beg + np.ceil(duration * 0.6).astype(np.int64)
    return multi_arange(idx_beg, idx_end)
