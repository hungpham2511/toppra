from .constraint import Constraint
from .constraint import ConstraintType, DiscretizationType

import numpy as np


class CanonicalLinearConstraint(Constraint):
    """  Base class for all canonical linear constraints.

    A canonical linear constraints can be written in the following form

    .. math::

        a[i] u + b[i] x + c[i] &= v, \\\\
        F[i] v &\\leq h[i], \\\\
        xbound[i, 0] \\leq x \\leq xbound[i, 1], \\\\
        ubound[i, 0] \\leq u \\leq ubound[i, 1].

    Derived classes should implement the method
    - compute_constraint_params(): tuple
    """
    def __init__(self):
        self.constraint_type = ConstraintType.CanonicalLinear
        self.discretization_type = DiscretizationType.Collocation
        self.n_extra_vars = 0

    def compute_constraint_params(self, path, gridpoints):
        """ Return constraint parameters.

        If a set of parameters are not available, None is returned.

        Parameters
        ----------
        path: `Interpolator`
            The geometric path.
        gridpoints: array
            (N+1,) array. The path discretization.

        Returns
        -------
        a: array, or None
            Shape (N + 1, m). See notes.
        b: array, or None
            Shape (N + 1, m). See notes.
        c: array, or None
            Shape (N + 1, m). See notes.
        F: array, or None
            Shape (N + 1, k, m). See notes.
        g: array, or None
            Shape (N + 1, k,). See notes
        ubound: array, or None
            Shape (N + 1, 2). See notes.
        xbound: array, or None
            Shape (N + 1, 2). See notes.

        """
        raise NotImplementedError


def canlinear_colloc_to_interpolate(a, b, c, F, g, xbound, ubound, gridpoints):
    """ Convert a set of parameters to the interpolation discretization scheme.

    If a set of parameters is None, the resulting set is also None.

    Parameters
    ----------
    a: array, or None
        Shape (N + 1, m). See notes.
    b: array, or None
        Shape (N + 1, m). See notes.
    c: array, or None
        Shape (N + 1, m). See notes.
    F: array, or None
        Shape (N + 1, k, m). See notes.
    g: array, or None
        Shape (N + 1, k,). See notes
    ubound: array, or None
        Shape (N + 1, 2). See notes.
    xbound: array, or None
        Shape (N + 1, 2). See notes.
    gridpoints: array
        (N+1,) array. The path discretization.

    Returns
    -------
    a_intp: array, or None
        Shape (N + 1, m). See notes.
    b_intp: array, or None
        Shape (N + 1, m). See notes.
    c_intp: array, or None
        Shape (N + 1, m). See notes.
    F_intp: array, or None
        Shape (N + 1, k, m). See notes.
    g_intp: array, or None
        Shape (N + 1, k,). See notes
    ubound: array, or None
        Shape (N + 1, 2). See notes.
    xbound: array, or None
        Shape (N + 1, 2). See notes.

    """
    if a is None:
        a_intp = None
        b_intp = None
        c_intp = None
        F_intp = None
        g_intp = None
    else:
        N = a.shape[0] - 1
        d = a.shape[1]
        m = g.shape[1]
        deltas = np.diff(gridpoints)

        a_intp = np.zeros((N + 1, 2 * d))
        a_intp[:, :d] = a
        a_intp[:-1, d:] = a[1:] + 2 * deltas.reshape(-1, 1) * b[1:]
        a_intp[-1, d:] = a_intp[-1, :d]

        b_intp = np.zeros((N + 1, 2 * d))
        b_intp[:, :d] = b
        b_intp[:-1, d:] = b[1:]
        b_intp[-1, d:] = b_intp[-1, :d]

        c_intp = np.zeros((N + 1, 2 * d))
        c_intp[:, :d] = c
        c_intp[:-1, d:] = c[1:]
        c_intp[-1, d:] = c_intp[-1, :d]

        g_intp = np.zeros((N + 1, 2 * m))
        g_intp[:, :m] = g
        g_intp[:-1, m:] = g[1:]
        g_intp[-1, m:] = g_intp[-1, :m]

        F_intp = np.zeros((N + 1, 2 * m, 2 * d))
        F_intp[:, :m, :d] = F
        F_intp[:-1, m:, d:] = F[1:]
        F_intp[-1, m:, d:] = F[-1]

    return a_intp, b_intp, c_intp, F_intp, g_intp, xbound, ubound

