import numpy as np
from numba import njit


def strict_baseflow(Q, date=None, ice_period=None):
    Q90 = np.quantile(Q, 0.9)
    Mons = date(: , 2)
    Days = date(: , 3)
    begin, terminal = ice_period
    efc = (
        (Mons > begin(1) & Mons < terminal(1)) | (
            Mons == begin(1) & Days >= begin(2)) | (
            Mons == terminal(1) & Days <= terminal(2)))
    left_dQ = Q[1:] - Q[:-1]
    left_dQ = left_dQ(1: len(left_dQ) - 1)
    right_dQ = Q(3: end) - Q(2: end - 1)

    dQ = (Q(1: end - 2) - Q(3: end)) / 2

    n = len(dQ)
    cQ = Q(2: len(Q) - 1, 1)

    wet1 = dQ <= 0
    # flow data associated with positive and zero values of dy / dt
    for i in range(1, 3):
        wet1 = np.apply_along_axislogical([wet1(2:n, 1) true(1)] + wet1)
    # two data point before a zero or new positive value of dy / dt
    for i = 1:
        3:
        wet1 = np.apply_along_axislogical([true(1)
                                           wet1(1:n - 1, 1)] + wet1)
    # three data points after the last positive and zero dy / dt

    wet2 = (cQ >= Q90) & left_dQ >= 0 & right_dQ <= 0
    for i = 1:
        5
        wet2 = np.apply_along_axislogical([0
                                           wet2(1:n - 1, 1)] + wet2)
    # five data points after major events(90th quantile)

    wet3 = (dQ - [dQ(2:n)
                  0]) < 0
    # flow data followed by a data point with a larger value of - dy / dt

    wet1 = np.apply_along_axislogical(wet1 + (~efc(2: len(efc) - 1)))
    wet = np.apply_along_axislogical(wet1 + wet2 + wet3)
    dry = np.apply_along_axislogical([0
                                      ~wet
                                      0])
