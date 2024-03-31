import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from baseflow.methods import *
from baseflow.comparision import strict_baseflow, KGE
from baseflow.utils import clean_streamflow, exist_ice, geo2imagexy, format_method
from baseflow.param_estimate import recession_coefficient, param_calibrate, maxmium_BFI


def single(series, area=None, ice=None, method='all', return_kge=True):
    Q, date = clean_streamflow(series)
    method = format_method(method)

    # convert ice_period ([11, 1], [3, 31]) to bool array
    if not isinstance(ice, np.ndarray) or ice.shape[0] == 12:
        ice = exist_ice(date, ice)
    strict = strict_baseflow(Q, ice)
    if any(m in ['Chapman', 'CM', 'Boughton', 'Furey', 'Eckhardt', 'Willems'] for m in method):
        a = recession_coefficient(Q, strict)

    b_LH = LH(Q)
    b = pd.DataFrame(np.nan, index=date, columns=method)
    for m in method:
        if m == 'UKIH':
            b[m] = UKIH(Q, b_LH)

        if m == 'Local':
            b[m] = Local(Q, b_LH, area)

        if m == 'Fixed':
            b[m] = Fixed(Q, area)

        if m == 'Slide':
            b[m] = Slide(Q, area)

        if m == 'LH':
            b[m] = b_LH

        if m == 'Chapman':
            b[m] = Chapman(Q, b_LH, a)

        if m == 'CM':
            b[m] = CM(Q, b_LH, a)

        if m == 'Boughton':
            C = param_calibrate(np.arange(0.0001, 0.1, 0.0001), Boughton, Q, b_LH, a)
            b[m] = Boughton(Q, b_LH, a, C)

        if m == 'Furey':
            A = param_calibrate(np.arange(0.01, 10, 0.01), Furey, Q, b_LH, a)
            b[m] = Furey(Q, b_LH, a, A)

        if m == 'Eckhardt':
            # BFImax = maxmium_BFI(Q, b_LH, a, date)
            BFImax = param_calibrate(np.arange(0.001, 1, 0.001), Eckhardt, Q, b_LH, a)
            b[m] = Eckhardt(Q, b_LH, a, BFImax)

        if m == 'EWMA':
            e = param_calibrate(np.arange(0.0001, 0.1, 0.0001), EWMA, Q, b_LH, 0)
            b[m] = EWMA(Q, b_LH, 0, e)

        if m == 'Willems':
            w = param_calibrate(np.arange(0.001, 1, 0.001), Willems, Q, b_LH, a)
            b[m] = Willems(Q, b_LH, a, w)

    if return_kge:
        KGEs = pd.Series(KGE(b[strict].values, np.repeat(
            Q[strict], len(method)).reshape(-1, len(method))), index=b.columns)
        return b, KGEs
    else:
        return b, None


def separation(df, df_sta=None, method='all', return_bfi=False, return_kge=False):
    # baseflow separation worker for single station
    def sep_work(s):
        try:
            # read area, longitude, latitude from df_sta
            area, ice = None, None
            to_num = lambda col: (pd.to_numeric(df_sta.loc[s, col], errors='coerce')
                                  if (df_sta is not None) and (col in df_sta.columns) else np.nan)
            if np.isfinite(to_num('area')):
                area = to_num('area')
            if np.isfinite(to_num('lon')):
                c, r = geo2imagexy(to_num('lon'), to_num('lat'))
                ice = ~thawed[:, r, c]
                ice = ([11, 1], [3, 31]) if ice.all() else ice
            # separate baseflow for station S
            b, KGEs = single(df[s], ice=ice, area=area, method=method, return_kge=return_kge)
            # write into already created dataframe
            for m in method:
                dfs[m].loc[b.index, s] = b[m]
            if return_bfi:
                df_bfi.loc[s] = b.sum() / df.loc[b.index, s].abs().sum()
            if return_kge:
                df_kge.loc[s] = KGEs
        except BaseException:
            pass

    # convert index to datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # thawed months from https://doi.org/10.5194/essd-9-133-2017
    with np.load(Path(__file__).parent / 'thawed.npz') as f:
        thawed = f['thawed']

    # create df to store baseflow
    method = format_method(method)
    dfs = {m: pd.DataFrame(np.nan, index=df.index, columns=df.columns, dtype=float)
           for m in method}

    # create df to store BFI and KGE
    if return_bfi:
        df_bfi = pd.DataFrame(np.nan, index=df.columns, columns=method, dtype=float)
    if return_kge:
        df_kge = pd.DataFrame(np.nan, index=df.columns, columns=method, dtype=float)

    # run separation for each column
    for s in tqdm(df.columns, total=df.shape[1]):
        sep_work(s)

    # return result
    if return_bfi and return_kge:
        return dfs, df_bfi, df_kge
    if return_bfi and not return_kge:
        return dfs, df_bfi
    if not return_bfi and return_kge:
        return dfs, df_kge
    return dfs
