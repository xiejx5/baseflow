import numpy as np
from numba import njit
from utils import logQNSE
from recession_analysis import recession_period


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
