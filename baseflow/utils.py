import numpy as np
from numba import njit


def clean_streamflow(series):
    date, Q = series.index, series.values.astype(float)
    has_value = np.isfinite(Q)
    date, Q = date[has_value], np.abs(Q[has_value])
    return date, Q


def exist_ice(date, ice_period):
    if (date is None) or (ice_period is None):
        return None

    if isinstance(ice_period, np.ndarray):
        return np.isin(date.month, np.where(ice_period)[0] + 1)

    beg, end = ice_period
    if (end[0] > beg[0]) or ((end[0] == beg[0]) & (end[1] > beg[1])):
        ice = (
            ((date.month > beg[0]) & (date.month < end[0]))
            | ((date.month == beg[0]) & (date.day >= beg[1]))
            | ((date.month == end[0]) & (date.day <= end[1]))
        )
    else:
        ice = (
            ((date.month > beg[0]) | (date.month < end[0]))
            | ((date.month == beg[0]) & (date.day >= beg[1]))
            | ((date.month == end[0]) & (date.day <= end[1]))
        )
    return ice


def moving_average(x, w):
    res = np.convolve(x, np.ones(w)) / w
    return res[w - 1 : -w + 1]


@njit
def multi_arange(starts, stops):
    pos = 0
    cnt = np.sum(stops - starts, dtype=np.int64)
    res = np.zeros((cnt,), dtype=np.int64)
    for i in range(starts.size):
        num = stops[i] - starts[i]
        res[pos : pos + num] = np.arange(starts[i], stops[i])
        pos += num
    return res


def geo2imagexy(x, y):
    a = np.array([[0.5, 0.0], [0.0, -0.5]])
    b = np.array([x - -180, y - 90])
    col, row = np.linalg.solve(a, b) - 0.5
    return np.round(col).astype(int), np.round(row).astype(int)


def format_method(method):
    if method == "all":
        method = [
            "UKIH",
            "Local",
            "Fixed",
            "Slide",
            "LH",
            "Chapman",
            "CM",
            "Boughton",
            "Furey",
            "Eckhardt",
            "EWMA",
            "Willems",
        ]
    elif isinstance(method, str):
        method = [method]
    return method
