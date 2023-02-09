import numpy as np
from numba import njit


@njit
def Eckhardt(Q, b_LH, a, BFImax, return_exceed=False):
    """Eckhardt filter (Eckhardt, 2005)

    Args:
        Q (np.array): streamflow
        a (float): recession coefficient
        BFImax (float): maximum value of baseflow index (BFI)
    """
    if return_exceed:
        b = np.zeros(Q.shape[0] + 1)
    else:
        b = np.zeros(Q.shape[0])
    b[0] = b_LH[0]
    for i in range(Q.shape[0] - 1):
        b[i + 1] = ((1 - BFImax) * a * b[i] + (1 - a) * BFImax * Q[i + 1]) / (1 - a * BFImax)
        if b[i + 1] > Q[i + 1]:
            b[i + 1] = Q[i + 1]
            if return_exceed:
                b[-1] += 1
    return b
