import numpy as np
from .constants import JVEL_MAXSD
cimport numpy as np
DTYPE = np.float64
ctypedef np.float64_t FLOAT_t

cdef inline np.float64_t float64_max(
    FLOAT_t a, FLOAT_t b): return a if a >= b else b
cdef inline np.float64_t float64_min(
    FLOAT_t a, FLOAT_t b): return a if a <= b else b
cdef inline np.float64_t float64_abs(
    FLOAT_t a): return a if a > 0 else - a

cdef float MAXSD = JVEL_MAXSD  # Maximum allowable path velocity for velocity constraint

cpdef _create_velocity_constraint(np.ndarray[double, ndim=2] qs,
                                  np.ndarray[double, ndim=2] vlim):
    """ Evaluate coefficient matrices for velocity constraints.

    A maximum allowable value for the path velocity is defined with
    `MAXSD`. This constant results in a lower bound on trajectory
    duration obtain by toppra.

    Args:
    ----
    qs: ndarray
        Path derivatives at each grid point.
    vlim: ndarray
        Velocity bounds.
    
    Returns:
    --------
    a: ndarray
    b: ndarray
    c: ndarray
        Coefficient matrices.
    """
    cdef int N = qs.shape[0] - 1
    cdef int dof = qs.shape[1]
    cdef int i, k
    cdef float sdmin, sdmax
    # Evaluate sdmin, sdmax at each steps and fill the matrices.
    cdef np.ndarray[np.float64_t, ndim=2] a = np.zeros((N + 1, 2), dtype=float)
    cdef np.ndarray[np.float64_t, ndim=2] b = np.ones((N + 1, 2), dtype=float)
    cdef np.ndarray[np.float64_t, ndim=2] c = np.zeros((N + 1, 2), dtype=float)
    b[:, 1] = -1
    for i in range(N + 1):
        sdmin = - MAXSD
        sdmax = MAXSD
        for k in range(dof):
            if qs[i, k] > 0:
                sdmax = float64_min(vlim[k, 1] / qs[i, k], sdmax)
                sdmin = float64_max(vlim[k, 0] / qs[i, k], sdmin)
            elif qs[i, k] < 0:
                sdmax = float64_min(vlim[k, 0] / qs[i, k], sdmax)
                sdmin = float64_max(vlim[k, 1] / qs[i, k], sdmin)
        c[i, 0] = - sdmax**2
        c[i, 1] = float64_max(sdmin, 0.)**2
    return a, b, c

cpdef _create_velocity_constraint_varying(np.ndarray[double, ndim=2] qs,
                                          np.ndarray[double, ndim=3] vlim_grid):
    """ Evaluate coefficient matrices for velocity constraints.

    Args:
    ----
    qs: (N,) ndarray
        Path derivatives at each grid point.
    vlim_grid: (N, dof, 2) ndarray
        Velocity bounds at each grid point.
    
    Returns:
    --------
    a: ndarray
    b: ndarray
    c: ndarray
        Coefficient matrices.
    """
    cdef int N = qs.shape[0] - 1
    cdef int dof = qs.shape[1]
    cdef int i, k
    cdef float sdmin, sdmax
    # Evaluate sdmin, sdmax at each steps and fill the matrices.
    cdef np.ndarray[np.float64_t, ndim=2] a = np.zeros((N + 1, 2), dtype=float)
    cdef np.ndarray[np.float64_t, ndim=2] b = np.ones((N + 1, 2), dtype=float)
    cdef np.ndarray[np.float64_t, ndim=2] c = np.zeros((N + 1, 2), dtype=float)
    b[:, 1] = -1
    for i in range(N + 1):
        sdmin = - MAXSD
        sdmax = MAXSD
        for k in range(dof):
            if qs[i, k] > 0:
                sdmax = float64_min(vlim_grid[i, k, 1] / qs[i, k], sdmax)
                sdmin = float64_max(vlim_grid[i, k, 0] / qs[i, k], sdmin)
            elif qs[i, k] < 0:
                sdmax = float64_min(vlim_grid[i, k, 0] / qs[i, k], sdmax)
                sdmin = float64_max(vlim_grid[i, k, 1] / qs[i, k], sdmin)
        c[i, 0] = - sdmax**2
        c[i, 1] = float64_max(sdmin, 0.)**2
    return a, b, c

