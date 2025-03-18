import numpy as np
from numba import njit, prange
from baseflow.methods.UKIH import linear_interpolation


def Local(Q, b_LH, area=None, return_exceed=False):
    """Local minimum graphical method from HYSEP program (Sloto & Crouse, 1996)

    Args:
        Q (np.array): streamflow
        area (float): basin area in km^2
    """
    idx_turn = Local_turn(Q, hysep_interval(area))
    if idx_turn.shape[0] < 3:
        raise IndexError("Less than 3 turning points found")
    b = linear_interpolation(Q, idx_turn, return_exceed=return_exceed)
    b[: idx_turn[0]] = b_LH[: idx_turn[0]]
    b[idx_turn[-1] + 1 :] = b_LH[idx_turn[-1] + 1 :]
    return b


def hysep_interval(area):
    # The duration of surface runoff is calculated from the empirical relation:
    # N=A^0.2, (1) where N is the number of days after which surface runoff ceases,
    # and A is the drainage area in square miles (Linsley and others, 1982, p. 210).
    # The interval 2N* used for hydrograph separations is the odd integer between
    # 3 and 11 nearest to 2N (Pettyjohn and Henning, 1979, p. 31).
    if area is None:
        N = 5
    else:
        N = np.power(0.3861022 * area, 0.2)
    inN = np.ceil(2 * N)
    if np.mod(inN, 2) == 0:
        inN = np.ceil(2 * N) - 1
    inN = np.int64(min(max(inN, 3), 11))
    return inN


@njit
def Local_turn(Q, inN):
    idx_turn = np.zeros(Q.shape[0], dtype=np.int64)
    for i in prange(np.int64((inN - 1) / 2), np.int64(Q.shape[0] - (inN - 1) / 2)):
        if Q[i] == np.min(Q[np.int64(i - (inN - 1) / 2) : np.int64(i + (inN + 1) / 2)]):
            idx_turn[i] = i
    return idx_turn[idx_turn != 0]
