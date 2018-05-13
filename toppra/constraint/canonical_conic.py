from .constraint import Constraint
from .constraint import ConstraintType, DiscretizationType
import numpy as np


class CanonicalConicConstraint(Constraint):
    """Base class for all canonical conic constraints.

    A canonical conic constraint has the following form

    .. math::

        (a[i] + da[i]) u + (b[i] + db[i]) x + (c[i] + dc[i]) \leq 0, \\\\
        [da[i, j], db[i, j], dc[i, j]]^\top = P[i, j] u, \|u\|_2 \leq 1,

    where P[i, j] is a 3x3 matrix.

    Constraints of this form is known mathematically as conic
    quadratic constraints. See [1] for more information.

    Refs:
    ----
    [1] Ben-Tal, A., & Nemirovski, A. (2001). Lectures on modern convex
        optimization: analysis, algorithms, and engineering applications
        (Vol. 2). Siam.
    """
    def __init__(self):
        self.constraint_type = ConstraintType.CanonicalConic
        self.discretization_type = DiscretizationType.Collocation
        self.n_extra_vars = 0
        self.dof = -1

    def compute_constraint_params(self, path, gridpoints):
        raise NotImplementedError


class RobustCanonicalLinearConstraint(CanonicalConicConstraint):
    """The simple canonical conic constraint.

    This constraint can be seen as a more robust version of a
    CanonicalLinear constraint. In particular, the perturbations term,
    [\Delta a[i, j], \Delta b[i, j], \Delta c[i, j]] is assumed to lie
    in a centered ellipsoid:

    .. math::
        [\Delta a[i, j], \Delta b[i, j], \Delta c[i, j]]^\top = diag(ru, rx, rc) \mathbf e,

    where \|\mathbf e\|_2 \leq 1.

    Parameters
    ----------
    cnst: `CanonicalLinearConstraint`
        The base CanonicalLinearConstraint.
    ellipsoid_axes_lengths: (3,)array
        Lengths of the axes of the perturbation ellipsoid. Must all be
        non-negative.
    """
    def __init__(self, cnst, ellipsoid_axes_lengths):
        super(RobustCanonicalLinearConstraint, self).__init__()
        self.dof = cnst.get_dof()
        assert cnst.get_constraint_type() == ConstraintType.CanonicalLinear
        if np.any(np.r_[ellipsoid_axes_lengths] < 0):
            raise ValueError("Perturbation must be non-negative. Input {:}".format(ellipsoid_axes_lengths))
        self.base_constraint = cnst
        self.ellipsoid_axes_lengths = ellipsoid_axes_lengths

    def compute_constraint_params(self, path, gridpoints):
        a_, b_, c_, F_, g_, _, _ = self.base_constraint.compute_constraint_params(path, gridpoints)

        a = np.zeros(F_.shape[:2])
        b = np.zeros(F_.shape[:2])
        c = np.zeros(F_.shape[:2])
        for i in range(len(gridpoints)):
            a[i] = F_[i].dot(a_[i])
            b[i] = F_[i].dot(b_[i])
            c[i] = F_[i].dot(c_[i]) - g_[i]

        P = np.zeros((F_.shape[0], F_.shape[1], 3, 3))
        diag_ = np.diag(self.ellipsoid_axes_lengths)
        for i in range(len(gridpoints)):
            P[i] = diag_
        return a, b, c, P
