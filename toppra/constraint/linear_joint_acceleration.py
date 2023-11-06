"""Joint acceleration constraint."""
import numpy as np
from .linear_constraint import LinearConstraint, canlinear_colloc_to_interpolate
from ..constraint import DiscretizationType
from ..interpolator import AbstractGeometricPath


class JointAccelerationConstraint(LinearConstraint):
    """The Joint Acceleration Constraint class.

    A joint acceleration constraint is given by

    .. math ::

                \ddot{\mathbf{q}}_{min} & \leq \ddot{\mathbf q}
                                                    &\leq \ddot{\mathbf{q}}_{max} \\\\
                \ddot{\mathbf{q}}_{min} & \leq \mathbf{q}'(s_i) u_i + \mathbf{q}''(s_i) x_i
                                                    &\leq \ddot{\mathbf{q}}_{max}

    where :math:`u_i, x_i` are respectively the path acceleration and
    path velocity square at :math:`s_i`. For more detail see :ref:`derivationKinematics`.

    Rearranging the above pair of vector inequalities into the form
    required by :class:`LinearConstraint`, we have:

    - :code:`a[i]` := :math:`\mathbf q'(s_i)`
    - :code:`b[i]` := :math:`\mathbf q''(s_i)`
    - :code:`F` := :math:`[\mathbf{I}, -\mathbf I]^T`
    - :code:`h` := :math:`[\ddot{\mathbf{q}}_{max}^T, -\ddot{\mathbf{q}}_{min}^T]^T`
    """

    def __init__(self, alim, discretization_scheme=DiscretizationType.Interpolation):
        """Initialize the joint acceleration class.

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
        super(JointAccelerationConstraint, self).__init__()
        alim = np.array(alim, dtype=float)
        if np.isnan(alim).any():
            raise ValueError("Bad acceleration given: %s" % alim)
        if len(alim.shape) == 1:
            self.alim = np.vstack((-np.array(alim), np.array(alim))).T
        else:
            self.alim = np.array(alim, dtype=float)
        self.dof = self.alim.shape[0]
        self.set_discretization_type(discretization_scheme)

        assert self.alim.shape[1] == 2, "Wrong input shape."
        self._format_string = "    Acceleration limit: \n"
        for i in range(self.alim.shape[0]):
            self._format_string += "      J{:d}: {:}".format(i + 1, self.alim[i]) + "\n"
        self.identical = True

    def compute_constraint_params(
        self, path: AbstractGeometricPath, gridpoints: np.ndarray, *args, **kwargs
    ):
        if path.dof != self.dof:
            raise ValueError(
                "Wrong dimension: constraint dof ({:d}) not equal to path dof ({:d})".format(
                    self.dof, path.dof
                )
            )
        ps_vec = (path(gridpoints, order=1)).reshape((-1, path.dof))
        pss_vec = (path(gridpoints, order=2)).reshape((-1, path.dof))
        dof = path.dof
        F_single = np.zeros((dof * 2, dof))
        g_single = np.zeros(dof * 2)
        g_single[0:dof] = self.alim[:, 1]
        g_single[dof:] = -self.alim[:, 0]
        F_single[0:dof, :] = np.eye(dof)
        F_single[dof:, :] = -np.eye(dof)
        if self.discretization_type == DiscretizationType.Collocation:
            return (
                ps_vec,
                pss_vec,
                np.zeros_like(ps_vec),
                F_single,
                g_single,
                None,
                None,
            )
        elif self.discretization_type == DiscretizationType.Interpolation:
            return canlinear_colloc_to_interpolate(
                ps_vec,
                pss_vec,
                np.zeros_like(ps_vec),
                F_single,
                g_single,
                None,
                None,
                gridpoints,
                identical=True,
            )
        else:
            raise NotImplementedError("Other form of discretization not supported!")


class JointAccelerationConstraintVarying(LinearConstraint):
    """A Joint acceleration Constraint class.

    This class handle acceleration constraints that vary along the path.

    Parameters
    ----------
    alim_func: (float) -> np.ndarray
        A function that receives a scalar (float) and produce an array
        with shape (dof, 2). The lower and upper acceleration bounds of
        the j-th joint are given by out[j, 0] and out[j, 1]
        respectively.
    """

    def __init__(self, alim_func):
        super(JointAccelerationConstraintVarying, self).__init__()
        self.dof = alim_func(0).shape[0]
        self._format_string = "    Varying Acceleration limit: \n"
        self.alim_func = alim_func
        self.identical = False

    def compute_constraint_params(self, path, gridpoints):
        if path.dof != self.dof:
            raise ValueError(
                "Wrong dimension: constraint dof ({:d}) not equal to path dof ({:d})".format(
                    self.dof, path.dof
                )
            )
        ps_vec = (path(gridpoints, order=1)).reshape((-1, path.dof))
        pss_vec = (path(gridpoints, order=2)).reshape((-1, path.dof))
        dof = path.dof
        n_grid = gridpoints.shape[0]

        stacked_eyes = np.vstack((np.eye(dof), -np.eye(dof)))
        f_vec = np.repeat(stacked_eyes[np.newaxis, :, :], n_grid, axis=0)

        # compute accel limits over gridpoints
        alim_grid = np.array([self.alim_func(s) for s in gridpoints])
        g_vec = np.zeros((n_grid, dof * 2))
        g_vec[:, 0:dof] = alim_grid[:, :, 1]
        g_vec[:, dof:] = -alim_grid[:, :, 0]

        if self.discretization_type == DiscretizationType.Collocation:
            return (
                ps_vec,
                pss_vec,
                np.zeros_like(ps_vec),
                f_vec,
                g_vec,
                None,
                None,
            )
        elif self.discretization_type == DiscretizationType.Interpolation:
            return canlinear_colloc_to_interpolate(
                ps_vec,
                pss_vec,
                np.zeros_like(ps_vec),
                f_vec,
                g_vec,
                None,
                None,
                gridpoints,
                identical=False,
            )
        else:
            raise NotImplementedError("Other form of discretization not supported!")
