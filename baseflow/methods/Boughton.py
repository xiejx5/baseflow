import numpy as np
from numba import njit


@njit
def Boughton(Q, a, C, return_exceed=False):
    """Boughton doulbe-parameter filter (Boughton, 2004)

    Args:
        Q (np.array): streamflow
        a (float): recession constant
        C (float): calibrated in baseflow.param_estimate
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
