from .canonical_linear import CanonicalLinearConstraint, canlinear_colloc_to_interpolate
from ..constraint import DiscretizationType
import numpy as np


class JointAccelerationConstraint(CanonicalLinearConstraint):
    """Joint Acceleration Constraint.

    A joint acceleration constraint is given by

    .. math ::

                \ddot{\mathbf{q}}_{min} & \leq \ddot{\mathbf q}                             &\leq \ddot{\mathbf{q}}_{max} \\\\
                \ddot{\mathbf{q}}_{min} & \leq \mathbf{q}'(s_i) u_i + \mathbf{q}''(s_i) x_i &\leq \ddot{\mathbf{q}}_{max}

    where :math:`u_i, x_i` are respectively the path acceleration and
    path velocity square at :math:`s_i`. For more detail see :ref:`derivationKinematics`.

    Rearranging the above pair of vector inequalities into the form
    required by :class:`CanonicalLinearConstraint`, we have:

    - :code:`a[i]` := :math:`\mathbf q'(s_i)`
    - :code:`b[i]` := :math:`\mathbf q''(s_i)`
    - :code:`F` := :math:`[\mathbf{I}, -\mathbf I]^T`
    - :code:`h` := :math:`[\ddot{\mathbf{q}}_{max}^T, -\ddot{\mathbf{q}}_{min}^T]^T`

    Parameters
    ----------
    alim: array
        Shape (dof, 2). The lower and upper acceleration bounds of the
        j-th joint are alim[j, 0] and alim[j, 1] respectively.

    discretization_scheme: :class:`.DiscretizationType`
        Can be either Collocation (0) or Interpolation
        (1). Interpolation gives more accurate results with slightly
        higher computational cost.

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

