import numpy as np
from numba import njit


@njit
def CM(Q, b_LH, a, return_exceed=False):
    """CM filter (Chapman & Maxwell, 1996)

    Args:
        Q (np.array): streamflow
        a (float): recession coefficient
    """
    if return_exceed:
        b = np.zeros(Q.shape[0] + 1)
    else:
        b = np.zeros(Q.shape[0])
    b[0] = b_LH[0]
    for i in range(Q.shape[0] - 1):
        b[i + 1] = a / (2 - a) * b[i] + (1 - a) / (2 - a) * Q[i + 1]
        if b[i + 1] > Q[i + 1]:
            b[i + 1] = Q[i + 1]
            if return_exceed:
                b[-1] += 1
    return b
