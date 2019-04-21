"""This module implements the general Second-Order constraints."""
import numpy as np
from .linear_constraint import LinearConstraint, canlinear_colloc_to_interpolate
from .constraint import DiscretizationType


class SecondOrderConstraint(LinearConstraint):
    """This class represents the linear generalized Second-order constraints.

    A `SecondOrderConstraint` is given by the following formula:

    .. math::
        A(\mathbf{q}) \ddot {\mathbf{q}} + \dot
        {\mathbf{q}}^\\top B(\mathbf{q}) \dot {\mathbf{q}} +
        C(\mathbf{q}) + sign(\dot {\mathbf{q}}) * D(\mathbf{q}) = w,

    where w is a vector that satisfies the polyhedral constraint:

    .. math::
        F(\mathbf{q}) w \\leq g(\mathbf{q}).

    The functions :math:`A, B, C, D` represent respectively the inertial,
    Corriolis, gravitational and dry friction terms in a robot torque
    bound constraint.

    To evaluate the constraint on a geometric path :math:`\mathbf{p}(s)`:

    .. math::

        A(\mathbf{q}) \mathbf{p}'(s) \ddot s + [A(\mathbf{q}) \mathbf{p}''(s) + \mathbf{p}'(s)^\\top B(\mathbf{q})
        \mathbf{p}'(s)] \dot s^2 + C(\mathbf{q}) + sign(\mathbf{p}'(s)) * D(\mathbf{p}(s)) = w, \\\\
        a(s) \ddot s + b(s) \dot s ^2 + c(s) = w.

    where :math:`\mathbf{p}', \mathbf{p}''` denote respectively the
    first and second derivatives of the path.


    It is important to note that to evaluate the coefficients
    :math:`a(s), b(s), c(s)`, it is not necessary to have the
    functions :math:`A, B, C`. Rather, only the sum of the these 3
    functions--the inverse dynamic function--is necessary.

    """

    def __init__(self, inv_dyn, cnst_F, cnst_g, dof, friction=None, discretization_scheme=1):
        """Initialize the constraint.

        Parameters
        ----------
        inv_dyn: [np.ndarray, np.ndarray, np.ndarray] -> np.ndarray
            The "inverse dynamics" function that receives joint
            position, velocity and acceleration as inputs and ouputs
            the "joint torque". It is not necessary to supply each
            individual component functions such as gravitational,
            Coriolis, etc.
        cnst_F: [np.ndarray] -> np.ndarray
            Coefficient function. See notes for more details.
        cnst_g: [np.ndarray] -> np.ndarray
            Coefficient function. See notes for more details.
        dof: int
            The dimension of the joint position.
        discretization_scheme: DiscretizationType
            Type of discretization.
        friction: [np.ndarray] -> np.ndarray
            Dry friction function.
        """
        super(SecondOrderConstraint, self).__init__()
        self.set_discretization_type(discretization_scheme)
        self.inv_dyn = inv_dyn
        self.cnst_F = cnst_F
        self.cnst_g = cnst_g
        self.dof = dof
        if friction is None:
            self.friction = lambda s: np.zeros(self.dof)
        else:
            self.friction = friction
        self._format_string = "    Kind: Generalized Second-order constraint\n"
        self._format_string = "    Dimension:\n"
        self._format_string += "        F in R^({:d}, {:d})\n".format(*cnst_F(np.zeros(dof)).shape)

    @staticmethod
    def joint_torque_constraint(inv_dyn, taulim, **kwargs):
        """Initialize a Joint Torque constraint.

        Parameters
        ----------
        inv_dyn: [np.ndarray, np.ndarray, np.ndarray] -> np.ndarray
            Inverse dynamic function of the robot.
        taulim: np.ndarray
            Shape (N, 2). The i-th element contains the minimum and maximum joint torque limits
            respectively.

        """
        dof = np.shape(taulim)[0]
        F_aug = np.vstack((np.eye(dof), - np.eye(dof)))
        g_aug = np.zeros(2 * dof)
        g_aug[:dof] = taulim[:, 1]
        g_aug[dof:] = - taulim[:, 0]
        cnst_F = lambda _: F_aug
        cnst_g = lambda _: g_aug
        return SecondOrderConstraint(inv_dyn, cnst_F, cnst_g, dof, **kwargs)

    def compute_constraint_params(self, path, gridpoints, scaling):
        if path.dof != self.dof:
            raise ValueError("Wrong dimension: constraint dof ({:d}) not equal to path dof ({:d})".format(
                self.dof, path.dof))
        v_zero = np.zeros(path.dof)
        p_vec = path.eval(gridpoints / scaling)
        ps_vec = path.evald(gridpoints / scaling) / scaling
        pss_vec = path.evaldd(gridpoints / scaling) / scaling ** 2

        F_vec = np.array(list(map(self.cnst_F, p_vec)))
        g_vec = np.array(list(map(self.cnst_g, p_vec)))
        c_vec = np.array([self.inv_dyn(_p, v_zero, v_zero) for _p in p_vec])
        a_vec = np.array([self.inv_dyn(_p, v_zero, _ps) for _p, _ps in zip(p_vec, ps_vec)]) - c_vec
        b_vec = np.array([self.inv_dyn(_p, _ps, pss_) for _p, _ps, pss_ in zip(p_vec, ps_vec, pss_vec)]) - c_vec

        for i, (_p, _ps) in enumerate(zip(p_vec, ps_vec)):
            c_vec[i] = c_vec[i] + np.sign(_ps) * self.friction(_p)

        if self.discretization_type == DiscretizationType.Collocation:
            return a_vec, b_vec, c_vec, F_vec, g_vec, None, None
        elif self.discretization_type == DiscretizationType.Interpolation:
            return canlinear_colloc_to_interpolate(a_vec, b_vec, c_vec, F_vec, g_vec, None, None, gridpoints)
        else:
            raise NotImplementedError("Other form of discretization not supported!")
