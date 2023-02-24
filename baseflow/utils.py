import numpy as np
from numba import njit


def load_streamflow(path):
    """load streamflow into memory

    Args:
        path (str|DataFrame): path of streamflow csv file, or pandas DataFrame

    Returns:
        tuple: (date of np.datetime64, streamflow of float)
    """
    if isinstance(path, str):
        date, Q = np.loadtxt(path, delimiter=',', skiprows=1, unpack=True,
                             dtype=[('date', 'datetime64[D]'), ('Q', float)],
                             converters={0: np.datetime64}, encoding='utf8')
        year = date.astype('datetime64[Y]').astype(int) + int(str(np.datetime64(0, 'Y')))
        month = date.astype('datetime64[M]').astype(int) % 12 + 1
        day = (date - date.astype('datetime64[M]')).astype(int) + 1
        date = np.rec.fromarrays([year, month, day], dtype=[('Y', 'i4'), ('M', 'i4'), ('D', 'i4')])
    elif len(path.shape) > 1:
        df_date = path.iloc[:, 0].astype('datetime64')
        date = np.rec.fromarrays([df_date.dt.year, df_date.dt.month, df_date.dt.day],
                                 dtype=[('Y', 'i4'), ('M', 'i4'), ('D', 'i4')])
        Q = path.iloc[:, 1].values.astype(float)
    else:
        date = np.rec.fromarrays([path.index.year, path.index.month, path.index.day],
                                 dtype=[('Y', 'i4'), ('M', 'i4'), ('D', 'i4')])
        Q = path.values.astype(float)
    return clean_streamflow(date, Q)


def clean_streamflow(date, Q):
    Q = np.abs(Q)
    year = date['Y']
    year_unique = np.unique(year)
    year_delete = clean_streamflow_jit(year, year_unique, Q)
    idx_delete = np.isin(year, year_delete) | np.isnan(Q)
    return Q[~idx_delete], date[~idx_delete]


@njit
def clean_streamflow_jit(year, year_unique, Q):
    year_delete = []
    for y in year_unique:
        if (Q[year == y] >= 0).sum() < 120:
            year_delete.append(y)
    return year_delete


def exist_ice(date, ice_period):
    if (date is None) or (ice_period is None):
        return None

    if isinstance(ice_period, np.ndarray):
        return np.isin(date.M, np.where(ice_period)[0] + 1)

    beg, end = ice_period
    if (end[0] > beg[0]) or ((end[0] == beg[0]) & (end[1] > beg[1])):
        ice = (((date.M > beg[0]) & (date.M < end[0])) |
               ((date.M == beg[0]) & (date.D >= beg[1])) |
               ((date.M == end[0]) & (date.D <= end[1])))
    else:
        ice = (((date.M > beg[0]) | (date.M < end[0])) |
               ((date.M == beg[0]) & (date.D >= beg[1])) |
               ((date.M == end[0]) & (date.D <= end[1])))
    return ice


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


@njit
def NSE(Q_obs, Q_sim):
    SS_res = np.sum(np.square(Q_obs - Q_sim))
    SS_tot = np.sum(np.square(Q_obs - np.mean(Q_obs)))
    return (1 - SS_res / (SS_tot + 1e-10)) - 1e-10
