from .canonical_linear import CanonicalLinearConstraint
from .._CythonUtils import _create_velocity_constraint
import numpy as np


class JointVelocityConstraint(CanonicalLinearConstraint):
    """ A Joint Velocity Constraint class.

    Parameters
    ----------
    vlim: array
        Shape (dof, 2). The lower and upper velocity bounds of the j-th joint
        are given by alim[j, 0] and alim[j, 1] respectively.

    """

    def __init__(self, vlim):
        super(JointVelocityConstraint, self).__init__()
        self.vlim = np.array(vlim, dtype=float)
        self.dof = self.vlim.shape[0]
        assert self.vlim.shape[1] == 2, "Wrong input shape."
        self._format_string = "    Velocity limit: \n"
        for i in range(self.vlim.shape[0]):
            self._format_string += "      J{:d}: {:}".format(i + 1, self.vlim[i]) + "\n"

    def compute_constraint_params(self, path, gridpoints):
        if path.get_dof() != self.get_dof():
            raise ValueError("Wrong dimension: constraint dof ({:d}) not equal to path dof ({:d})".format(
                self.get_dof(), path.get_dof()
            ))
        qs = path.evald(gridpoints)
        _, _, xbound_ = _create_velocity_constraint(qs, self.vlim)
        xbound = np.array(xbound_)
        xbound[:, 0] = xbound_[:, 1]
        xbound[:, 1] = - xbound_[:, 0]
        return None, None, None, None, None, None, xbound