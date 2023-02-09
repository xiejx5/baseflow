import numpy as np
from baseflow.methods import *
from baseflow.comparision import strict_baseflow, KGE
from baseflow.param_estimate import recession_coefficient, param_calibrate


def separation(Q, date=None, area=None, ice_period=None, method='all'):
    Q = np.array(Q)

    if method == 'all':
        method = ['UKIH', 'Local', 'Fixed', 'Slide', 'LH', 'Chapman',
                  'CM', 'Boughton', 'Furey', 'Eckhardt', 'EWMA', 'Willems']
    elif isinstance(method, str):
        method = [method]

    strict = strict_baseflow(Q)
    if any(m in ['Chapman', 'CM', 'Boughton', 'Furey', 'Eckhardt'] for m in method):
        a = recession_coefficient(Q, strict, date, ice_period)

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
            C = param_calibrate(np.arange(0.0001, 1, 0.0001), f_Boughton(a), Q, b_LH)
            b[m] = Boughton(Q, b_LH, a, C)

        if m == 'Furey':
            A = param_calibrate(np.arange(0.001, 10, 0.001), f_Furey(a), Q, b_LH)
            b[m] = Furey(Q, b_LH, a, A)

        if m == 'Eckhardt':
            # BFImax = maxmium_BFI(Q, b_LH, a, date)
            BFImax = param_calibrate(np.arange(0.0001, 1, 0.0001), f_Eckhardt(a), Q, b_LH)
            b[m] = Eckhardt(Q, b_LH, a, BFImax)

        if m == 'EWMA':
            e = param_calibrate(np.arange(0.0001, 0.5, 0.0001), EWMA, Q, b_LH)
            b[m] = EWMA(Q, b_LH, e)

        if m == 'Willems':
            w = param_calibrate(np.arange(0.0001, 1, 0.0001), f_Willems(a), Q, b_LH)
            b[m] = Willems(Q, b_LH, a, w)

    KGEs = KGE(b[strict].view(np.float64).reshape(-1, len(method)),
               np.repeat(Q[strict], len(method)).reshape(-1, len(method)))
    return b, KGEs
