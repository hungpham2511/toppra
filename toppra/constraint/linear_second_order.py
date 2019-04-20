from .linear_constraint import LinearConstraint, canlinear_colloc_to_interpolate
from .constraint import DiscretizationType
import numpy as np


class SecondOrderConstraint(LinearConstraint):
    """A class to represent Canonical Linear Generalized Second-order constraints.

    Notes
    -----
    A Second Order Constraint can be given by the following formula:

    .. math::
        A(q) \ddot q + \dot q^\\top B(q) \dot q + C(q) + sign(\dot q) * D(q) = w,

    where w is a vector that satisfies the polyhedral constraint:

    .. math::
        F(q) w \\leq g(q).

    The functions `A, B, C, D` can represent respectively the
    inertial, Corriolis, gravitational and dry friction term for robot
    torque bound constraint.

    To evaluate the constraint on a geometric path `p(s)`, multiple
    calls to `inv_dyn` and `const_coeff` are made as follows:

    .. math::

        A(q) p'(s) \ddot s + [A(q) p''(s) + p'(s)^\\top B(q) p'(s)] \dot s^2 + C(q) + sign(p'(s)) * D(p(s)) = w, \\\\
        a(s) \ddot s + b(s) \dot s ^2 + c(s) = w.

    To evaluate the coefficients a(s), b(s), c(s), inv_dyn is called
    repeatedly with appropriate arguments.

    """

    def __init__(self, inv_dyn, cnst_F, cnst_g, dof, discretization_scheme=1, friction=None):
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
        F_ = cnst_F(np.zeros(dof))
        self._format_string += "        F in R^({:d}, {:d})\n".format(*F_.shape)

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

        F = np.array(list(map(self.cnst_F, p_vec)))
        g = np.array(list(map(self.cnst_g, p_vec)))
        c = np.array([self.inv_dyn(p_, v_zero, v_zero) for p_ in p_vec])
        a = np.array([self.inv_dyn(p_, v_zero, ps_) for p_, ps_ in zip(p_vec, ps_vec)]) - c
        b = np.array([self.inv_dyn(p_, ps_, pss_) for p_, ps_, pss_ in zip(p_vec, ps_vec, pss_vec)]) - c

        for i, (p_, ps_) in enumerate(zip(p_vec, ps_vec)):
            c[i] = c[i] + np.sign(ps_) * self.friction(p_)

        if self.discretization_type == DiscretizationType.Collocation:
            return a, b, c, F, g, None, None
        elif self.discretization_type == DiscretizationType.Interpolation:
            return canlinear_colloc_to_interpolate(a, b, c, F, g, None, None, gridpoints)
        else:
            raise NotImplementedError("Other form of discretization not supported!")
