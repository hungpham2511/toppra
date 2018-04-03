import numpy as np

from .constraint import Constraint
from .constraint import ConstraintType
from .._CythonUtils import _create_velocity_constraint
from ..constants import MAXX, MAXU, TINY, SUPERTINY


class CanonicalLinearConstraint(Constraint):

    def get_constraint_type(self):
        return ConstraintType.CanonicalLinear

    def get_constraint_params(self, path, ss):
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


class JointVelocityConstraint(CanonicalLinearConstraint):
    """ A Joint Velocity Constraint class.

    Parameters
    ----------
    vlim: array
        Shape (dof, 2). The lower and upper velocity bounds of the j-th joint
        are given by alim[j, 0] and alim[j, 1] respectively.

    """

    def __init__(self, vlim):
        self.vlim = np.array(vlim)
        assert self.vlim.shape[1] == 2, "Wrong input shape."
        self._format_string = "    Velocity limit: \n" + str(self.vlim)

    def get_constraint_params(self, path, ss):
        qs = path.evald(ss)
        # Return resulti from cython version
        _, _, xbound_ = _create_velocity_constraint(qs, self.vlim)
        xbound = np.array(xbound_)
        xbound[:, 0] = xbound_[:, 1]
        xbound[:, 1] = - xbound_[:, 0]
        return None, None, None, None, None, None, xbound


class JointAccelerationConstraint(CanonicalLinearConstraint):
    """ A Joint Acceleration Constraint class.

    Joint acceleration constraint is given by

    .. math ::
                qdd_{min} \leq q'(s[i]) u[i] + q''(s[i]) x[i] \leq qdd_{max}

    Parameters
    ----------
    alim: array
        Shape (dof, 2). The lower and upper acceleration bounds of the j-th joint
        are given by alim[j, 0] and alim[j, 1] respectively.

    Notes
    -----
    The non-None constraint parameters output by this class are

    - `a` := q'(s)
    - `b` := q''(s)
    - `F` := [I, -I]^T
    - `b` := [qdd_max, -qdd_min]
    - `ubound` := [-UMAX, UMAX]

    """
    def __init__(self, alim):
        self.alim = np.array(alim)
        assert self.alim.shape[1] == 2, "Wrong input shape."
        self._format_string = "    Acceleration limit: \n" + str(self.alim)

    def get_constraint_params(self, path, ss):
        ps = path.evald(ss)
        pss = path.evaldd(ss)
        N = ss.shape[0] - 1
        dof = path.get_dof()
        I_dof = np.eye(dof)
        F = np.zeros((N + 1, dof * 2, dof))
        g = np.zeros((N + 1, dof * 2))
        ubound = np.zeros((N + 1, 2))
        g[:, 0:dof] = self.alim[:, 1]
        g[:, dof:] = - self.alim[:, 0]
        F[:, 0:dof, :] = I_dof
        F[:, dof:, :] = -I_dof
        ubound[:, 0] = - MAXU
        ubound[:, 1] = MAXU

        return ps, pss, None, F, g, ubound, None