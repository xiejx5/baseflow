import numpy as np
from numba import njit, prange
from baseflow.methods.Local import hysep_interval


def Fixed(Q, area=None):
    """Fixed interval graphical method from HYSEP program (Sloto & Crouse, 1996)

    Args:
        Q (np.array): streamflow
        area (float): basin area in km^2
    """
    inN = hysep_interval(area)
    return Fixed_interpolation(Q, inN)


@njit
def Fixed_interpolation(Q, inN):
    b = np.zeros(Q.shape[0])
    n = Q.shape[0] // inN
    for i in prange(n):
        b[inN * i : inN * (i + 1)] = np.min(Q[inN * i : inN * (i + 1)])
    if n * inN != Q.shape[0]:
        b[n * inN :] = np.min(Q[n * inN :])
    return b
