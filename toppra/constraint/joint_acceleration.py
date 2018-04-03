from .canonical_linear import CanonicalLinearConstraint
from ..constants import MAXU
import numpy as np


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
        self._format_string = "    Acceleration limit: \n"
        for i in range(self.alim.shape[0]):
            self._format_string += "      J{:d}: {:}".format(i + 1, self.alim[i]) + "\n"

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
