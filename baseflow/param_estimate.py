import numpy as np
from numba import njit
from baseflow.utils import logQNSE, moving_average, multi_arange


@njit
def recession_constant(Q, ice_period):
    return 0.98


@njit
def BFI_maxmium(Q, date):
    return 0.9


@njit
def param_calibrate(p_range, method, Q):
    idx_rec = recession_period(Q)
    loss = np.zeros(p_range.shape)
    for i, p in enumerate(p_range):
        b_exceed = method(Q, p, return_exceed=True)
        f_exd, b = b_exceed[-1] / Q.shape[0], b_exceed[:-1]
        idx_oth = np.delete(np.arange(Q.shape[0]), idx_rec)
        NSE_rec = logQNSE(Q[idx_rec], b[idx_rec])
        NSE_oth = logQNSE(Q[idx_oth], b[idx_oth])
        loss[i] = 1 - (1 - (1 - NSE_rec) / (1 - NSE_oth)) * (1 - f_exd)
    return p_range[np.argmin(loss)]


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
