import numpy as np
from numba import njit, prange
from baseflow.utils import moving_average, multi_arange


def recession_coefficient(Q, strict):
    cQ, dQ = Q[1:-1], (Q[2:] - Q[:-2]) / 2
    cQ, dQ = cQ[strict[1:-1]], dQ[strict[1:-1]]

    idx = np.argsort(-dQ / cQ)[np.floor(dQ.shape[0] * 0.05).astype(int)]
    K = -cQ[idx] / dQ[idx]
    return np.exp(-1 / K)


def param_calibrate(param_range, method, Q, b_LH, a):
    idx_rec = recession_period(Q)
    idx_oth = np.full(Q.shape[0], True)
    idx_oth[idx_rec] = False
    return param_calibrate_jit(param_range, method, Q, b_LH, a, idx_rec, idx_oth)


@njit(parallel=True)
def param_calibrate_jit(param_range, method, Q, b_LH, a, idx_rec, idx_oth):
    logQ = np.log1p(Q)
    loss = np.zeros(param_range.shape)
    for i in prange(param_range.shape[0]):
        p = param_range[i]
        b_exceed = method(Q, b_LH, a, p, return_exceed=True)
        f_exd, logb = b_exceed[-1] / Q.shape[0], np.log1p(b_exceed[:-1])

        # NSE for recession part
        Q_obs, Q_sim = logQ[idx_rec], logb[idx_rec]
        SS_res = np.sum(np.square(Q_obs - Q_sim))
        SS_tot = np.sum(np.square(Q_obs - np.mean(Q_obs)))
        NSE_rec = (1 - SS_res / (SS_tot + 1e-10)) - 1e-10

        # NSE for other part
        Q_obs, Q_sim = logQ[idx_oth], logb[idx_oth]
        SS_res = np.sum(np.square(Q_obs - Q_sim))
        SS_tot = np.sum(np.square(Q_obs - np.mean(Q_obs)))
        NSE_oth = (1 - SS_res / (SS_tot + 1e-10)) - 1e-10

        loss[i] = 1 - (1 - (1 - NSE_rec) / (1 - NSE_oth)) * (1 - f_exd)
    return param_range[np.argmin(loss)]


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


def maxmium_BFI(Q, b_LH, a, date=None):
    b = Backward(Q, b_LH, a)

    if date is None:
        idx_end = b.shape[0] // 365 * 365
        annual_b = np.mean(b[:idx_end].reshape(-1, 365), axis=1)
        annual_Q = np.mean(Q[:idx_end].reshape(-1, 365), axis=1)
        annual_BFI = annual_b / annual_Q
    else:
        idx_year = date.year - date.year.min()
        counts = np.bincount(idx_year)
        idx_valid = counts > 0
        annual_b = np.bincount(idx_year, weights=b)[idx_valid] / counts[idx_valid]
        annual_Q = np.bincount(idx_year, weights=Q)[idx_valid] / counts[idx_valid]
        annual_BFI = annual_b / annual_Q

    BFI_max = np.max(annual_BFI)
    BFI_max = BFI_max if BFI_max < 0.9 else np.sum(annual_b) / np.sum(annual_Q)
    return BFI_max


@njit
def Backward(Q, b_LH, a):
    b = np.zeros(Q.shape[0])
    b[-1] = b_LH[-1]
    for i in range(Q.shape[0] - 1, 0, -1):
        b[i - 1] = b[i] / a
        if b[i] == 0:
            b[i - 1] = Q[i - 1]
        if b[i - 1] > Q[i - 1]:
            b[i - 1] = Q[i - 1]
    return b
