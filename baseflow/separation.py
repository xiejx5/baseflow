import numpy as np
from baseflow.methods import *
from baseflow.recession_analysis import recession_constant
from baseflow.param_estimate import param_calibrate, BFI_maxmium


def separation(Q, date=None, area=None, ice_period=None, method='all'):
    if method == 'all':
        method = ['UKIH', 'Local', 'Fixed', 'Slide', 'LH', 'Chapman',
                  'CM', 'Boughton', 'Furey', 'Eckhardt', 'EWMA', 'Willems']
    elif isinstance(method, str):
        method = [method]

    if any(m in ['Chapman', 'CM', 'Boughton', 'Furey', 'Eckhardt'] for m in method):
        a = recession_constant(Q, ice_period)

    b = np.zeros([Q.shape[0], len(method)])
    i = 0
    for m in method:
        if m == 'UKIH':
            b[:, i] = UKIH(Q)

        if m == 'Local':
            b[:, i] = Local(Q, area)

        if m == 'Fixed':
            b[:, i] = Fixed(Q, area)

        if m == 'Slide':
            b[:, i] = Slide(Q, area)

        if m == 'LH':
            b[:, i] = LH(Q)

        if m == 'Chapman':
            b[:, i] = Chapman(Q, a)

        if m == 'CM':
            b[:, i] = CM(Q, a)

        if m == 'Boughton':
            C = param_calibrate(np.arange(0.0001, 1, 0.0001), Boughton, Q)
            b[:, i] = Boughton(Q, a, C)

        if m == 'Furey':
            A = param_calibrate(np.arange(0.001, 10, 0.001), Furey, Q)
            b[:, i] = Furey(Q, a, A)

        if m == 'Eckhardt':
            BFImax = BFI_maxmium(Q, date)
            b[:, i] = Eckhardt(Q, a, BFImax)

        if m == 'EWMA':
            e = param_calibrate(np.arange(0.0001, 0.5, 0.0001), EWMA, Q)
            b[:, i] = EWMA(Q, e)

        if m == 'Willems':
            w = param_calibrate(np.arange(0.0001, 1, 0.0001), Willems, Q)
            b[:, i] = Willems(Q, a, w)

    return b
