from .constraint import Constraint
from .constraint import ConstraintType


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
        super(CanonicalLinearConstraint, self).__init__()
        self.constraint_type = ConstraintType.CanonicalLinear

    def compute_constraint_params(self, path, ss):
        """ Return constraint parameter for Canonical Linear Constraints.

        Parameters
        ----------
        path: `Interpolator`
            The geometric path.
        ss: array
            The path discretization.

        Returns
        -------
        a: array
            Shape (N, m). See notes.
        b: array
            Shape (N, m). See notes.
        c: array
            Shape (N, m). See notes.
        F: array
            Shape (N, k, m). See notes.
        g: array
            Shape (N, k,). See notes
        ubound: array, or None
            Shape (N, 2). See notes.
        xbound: array, or None
            Shape (N, 2). See notes.

        Notes
        -----
        The general canonical linear constraint has this form

        .. math::

            a[i] u_1 + b[i] x + c[i] &= v, \\\\
            F[i] v &\\leq g[i], \\\\
            xbound[i, 0] \\leq x \\leq xbound[i, 1], \\\\
            ubound[i, 0] \\leq u \\leq ubound[i, 1].

        """
        raise NotImplementedError
