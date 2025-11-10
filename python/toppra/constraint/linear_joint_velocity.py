"""This module implements the joint velocity constraint."""
import numpy as np
from toppra._CythonUtils import (_create_velocity_constraint,
                                 _create_velocity_constraint_varying)
from .linear_constraint import LinearConstraint


class JointVelocityConstraint(LinearConstraint):
    """A Joint Velocity Constraint class.

    Parameters
    ----------
    vlim: np.ndarray
        Shape (dof, 2). The lower and upper velocity bounds of the j-th joint
        are given by vlim[j, 0] and vlim[j, 1] respectively.

    """

    def __init__(self, vlim):
        super(JointVelocityConstraint, self).__init__()
        vlim = np.array(vlim, dtype=float)
        if np.isnan(vlim).any():
            raise ValueError("Bad velocity given: %s" % vlim)
        if len(vlim.shape) == 1:
            self.vlim = np.vstack((-np.array(vlim), np.array(vlim))).T
        else:
            self.vlim = np.array(vlim, dtype=float)
        self.dof = self.vlim.shape[0]
        self._assert_valid_limits()

    def _assert_valid_limits(self):
        """Check that the velocity limits is valid."""
        assert self.vlim.shape[1] == 2, "Wrong input shape."
        for i in range(self.dof):
            if self.vlim[i, 0] >= self.vlim[i, 1]:
                raise ValueError("Bad velocity limits: {:} (lower limit) > {:} (higher limit)".format(
                    self.vlim[i, 0], self.vlim[i, 1]))
        self._format_string = "    Velocity limit: \n"
        for i in range(self.vlim.shape[0]):
            self._format_string += "      J{:d}: {:}".format(
                i + 1, self.vlim[i]) + "\n"

    def compute_constraint_params(self, path, gridpoints):
        if path.dof != self.get_dof():
            raise ValueError(
                "Wrong dimension: constraint dof ({:d}) not equal to path dof ({:d})"
                .format(self.get_dof(), path.dof))
        qs = path(gridpoints, 1)
        _, _, xbound_ = _create_velocity_constraint(qs, self.vlim)
        xbound = np.array(xbound_)
        xbound[:, 0] = xbound_[:, 1]
        xbound[:, 1] = -xbound_[:, 0]
        return None, None, None, None, None, None, xbound


class JointVelocityConstraintVarying(LinearConstraint):
    """A Joint Velocity Constraint class.

    This class handle velocity constraints that vary along the path.

    Parameters
    ----------
    vlim_func: (float) -> np.ndarray
        A function that receives a scalar (float) and produce an array
        with shape (dof, 2). The lower and upper velocity bounds of
        the j-th joint are given by out[j, 0] and out[j, 1]
        respectively.
    """

    def __init__(self, vlim_func):
        super(JointVelocityConstraintVarying, self).__init__()
        self.dof = vlim_func(0).shape[0]
        self._format_string = "    Varying Velocity limit: \n"
        self.vlim_func = vlim_func

    def compute_constraint_params(self, path, gridpoints):
        if path.dof != self.get_dof():
            raise ValueError(
                "Wrong dimension: constraint dof ({:d}) not equal to path dof ({:d})"
                .format(self.get_dof(), path.dof))
        qs = path((gridpoints), 1)
        vlim_grid = np.array([self.vlim_func(s) for s in gridpoints])
        _, _, xbound_ = _create_velocity_constraint_varying(qs, vlim_grid)
        xbound = np.array(xbound_)
        xbound[:, 0] = xbound_[:, 1]
        xbound[:, 1] = -xbound_[:, 0]
        return None, None, None, None, None, None, xbound
