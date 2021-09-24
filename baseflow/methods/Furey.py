import numpy as np
from numba import njit


@njit
def Furey(Q, a, A, return_exceed=False):
    """Furey digital filter (Furey & Gupta, 2001, 2003)

    Args:
        Q (np.array): streamflow
        a (float): recession constant
        A (float): calibrated in baseflow.param_estimate
    """
    if return_exceed:
        b = np.zeros(Q.shape[0] + 1)
    else:
        b = np.zeros(Q.shape[0])
    b[0] = Q[0]
    for i in range(Q.shape[0] - 1):
        b[i + 1] = (1 - e) * b[i] + e * Q[i + 1]
        if b[i + 1] > Q[i + 1]:
            b[i + 1] = Q[i + 1]
            if return_exceed:
                b[-1] += 1
    return b
