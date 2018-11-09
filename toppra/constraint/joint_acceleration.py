from .canonical_linear import CanonicalLinearConstraint, canlinear_colloc_to_interpolate
from ..constraint import DiscretizationType
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
    """
    def __init__(self, alim, discretization_scheme=DiscretizationType.Collocation):
        super(JointAccelerationConstraint, self).__init__()
        self.alim = np.array(alim, dtype=float)
        self.dof = self.alim.shape[0]
        self.set_discretization_type(discretization_scheme)
        assert self.alim.shape[1] == 2, "Wrong input shape."
        self._format_string = "    Acceleration limit: \n"
        for i in range(self.alim.shape[0]):
            self._format_string += "      J{:d}: {:}".format(i + 1, self.alim[i]) + "\n"
        self.identical = True

    def compute_constraint_params(self, path, gridpoints, scaling):
        if path.get_dof() != self.get_dof():
            raise ValueError("Wrong dimension: constraint dof ({:d}) not equal to path dof ({:d})".format(
                self.get_dof(), path.get_dof()
            ))
        ps = path.evald(gridpoints / scaling) / scaling
        pss = path.evaldd(gridpoints / scaling) / scaling ** 2
        N = gridpoints.shape[0] - 1
        dof = path.get_dof()
        I_dof = np.eye(dof)
        F = np.zeros((dof * 2, dof))
        g = np.zeros(dof * 2)
        ubound = np.zeros((N + 1, 2))
        g[0:dof] = self.alim[:, 1]
        g[dof:] = - self.alim[:, 0]
        F[0:dof, :] = I_dof
        F[dof:, :] = -I_dof
        if self.discretization_type == DiscretizationType.Collocation:
            return ps, pss, np.zeros_like(ps), F, g, None, None
        elif self.discretization_type == DiscretizationType.Interpolation:
            return canlinear_colloc_to_interpolate(ps, pss, np.zeros_like(ps), F, g, None, None,
                                                   gridpoints, identical=True)
        else:
            raise NotImplementedError("Other form of discretization not supported!")

