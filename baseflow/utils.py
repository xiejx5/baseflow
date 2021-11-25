import numpy as np
from numba import njit


def load_streamflow(path):
    """load streamflow into memory

    Args:
        path (str): path of streamflow csv file

    Returns:
        tuple: (date of np.datetime64, streamflow of float)
    """
    date, Q = np.loadtxt(path, delimiter=',', skiprows=1, unpack=True,
                         dtype=[('date', 'datetime64[D]'), ('Q', float)],
                         converters={0: np.datetime64})
    year = date.astype('datetime64[Y]').astype(int) + int(str(np.datetime64(0, 'Y')))
    month = date.astype('datetime64[M]').astype(int) % 12 + 1
    day = (date - date.astype('datetime64[M]')).astype(int) + 1
    date = np.rec.fromarrays([year, month, day], dtype=[('Y', 'i4'), ('M', 'i4'), ('D', 'i4')])
    return clean_streamflow(date, Q)


def clean_streamflow(date, Q):
    Q[np.isnan(Q)] = 0
    Q = np.abs(Q)
    year = date['Y']
    year_unique = np.unique(year)
    year_delete = clean_streamflow_jit(year, year_unique, Q)
    idx_delete = np.isin(year, year_delete)
    return Q[~idx_delete], date[~idx_delete]


@njit
def clean_streamflow_jit(year, year_unique, Q):
    year_delete = []
    for y in year_unique:
        if (Q[year == y] >= 0).sum() < 120:
            year_delete.append(y)
    return year_delete


@njit
def moving_average(x, w):
    res = np.convolve(x, np.ones(w)) / w
    return res[w - 1:-w + 1]


@njit
def multi_arange_steps(starts, stops, steps):
    pos = 0
    cnt = np.sum((stops - starts + steps - np.sign(steps)) // steps, dtype=np.int64)
    res = np.zeros((cnt,), dtype=np.int64)
    for i in range(starts.size):
        v, stop, step = starts[i], stops[i], steps[i]
        if step > 0:
            while v < stop:
                res[pos] = v
                pos += 1
                v += step
        elif step < 0:
            while v > stop:
                res[pos] = v
                pos += 1
                v += step
    assert pos == cnt
    return res


@njit
def multi_arange(starts, stops):
    pos = 0
    cnt = np.sum(stops - starts, dtype=np.int64)
    res = np.zeros((cnt,), dtype=np.int64)
    for i in range(starts.size):
        num = stops[i] - starts[i]
        res[pos:pos + num] = np.arange(starts[i], stops[i])
        pos += num
    return res


def NSE(Q_obs, Q_sim):
    SS_res = np.sum(np.square(Q_obs - Q_sim))
    SS_tot = np.sum(np.square(Q_obs - np.mean(Q_obs)))
    return (1 - SS_res / (SS_tot + np.finfo(float).eps))


@njit
def logQNSE(Q_obs, Q_sim):
    Q_obs, Q_sim = np.clip(Q_obs, 1e-10, None), np.clip(Q_sim, 1e-10, None)
    Q_obs, Q_sim = np.log10(Q_obs), np.log10(Q_sim)
    SS_res = np.sum(np.square(Q_obs - Q_sim))
    SS_tot = np.sum(np.square(Q_obs - np.mean(Q_obs)))
    return (1 - SS_res / (SS_tot + 1e-10)) - 1e-10
