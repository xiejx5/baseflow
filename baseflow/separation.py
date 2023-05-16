import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from baseflow.methods import *
from baseflow.comparision import strict_baseflow, KGE
from baseflow.utils import load_streamflow, exist_ice, geo2imagexy, format_method
from baseflow.param_estimate import recession_coefficient, param_calibrate, maxmium_BFI


def separation(Q, date=None, area=None, ice=None, method='all', return_kge=True):
    Q = np.array(Q)
    method = format_method(method)

    # convert ice_period ([11, 1], [3, 31]) to bool array
    if not isinstance(ice, np.ndarray) or ice.shape[0] == 12:
        ice = exist_ice(date, ice)
    strict = strict_baseflow(Q, ice)
    if any(m in ['Chapman', 'CM', 'Boughton', 'Furey', 'Eckhardt', 'Willems'] for m in method):
        a = recession_coefficient(Q, strict)

    b_LH = LH(Q)
    b = np.recarray(Q.shape[0], dtype=list(zip(method, [float] * len(method))))
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
        KGEs = KGE(b[strict].view(np.float64).reshape(-1, len(method)),
                   np.repeat(Q[strict], len(method)).reshape(-1, len(method)))
        return b, KGEs
    else:
        return b, None


def index(df, df_sta, method='all', return_kge=False):
    # baseflow index worker
    def index_work(idx):
        try:
            c, r = geo2imagexy(df_sta.loc[idx, 'lon'], df_sta.loc[idx, 'lat'])
            ice_period = ~thawed[:, r, c]
            ice_period = ([11, 1], [3, 31]) if ice_period.all() else ice_period
            Q, date = load_streamflow(df[idx])
            ice = exist_ice(date, ice_period)
            b, KGEs = separation(Q, ice=ice, area=df_sta.loc[idx, 'area'],
                                 method=method, return_kge=return_kge)
            df_bfi.loc[idx] = pd.DataFrame(b).sum() / Q.sum()
            if KGEs is not None:
                df_kge.loc[idx] = KGEs
        except BaseException:
            pass

    # read thawed months
    with np.load(Path(__file__).parent / 'thawed.npz') as f:
        thawed = f['thawed']

    # create df to store baseflow index
    method = format_method(method)
    df_bfi = pd.DataFrame(-1, index=df_sta.index, columns=method, dtype=float)

    # create df to store KGE
    if return_kge:
        df_kge = pd.DataFrame(-1, index=df_sta.index, columns=method, dtype=float)

    # run separation for each column
    for idx in tqdm(df_sta.index, total=df_sta.shape[0]):
        index_work(idx)

    # return result
    if return_kge:
        return df_bfi, df_kge
    else:
        return df_bfi
