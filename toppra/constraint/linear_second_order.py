from .linear_constraint import LinearConstraint, canlinear_colloc_to_interpolate
from .constraint import DiscretizationType
import numpy as np


class SecondOrderConstraint(LinearConstraint):
    """A class to represent Canonical Linear Generalized Second-order constraints.

    Notes
    -----
    A Second Order Constraint can be given by the following formula:

    .. math::
        A(q) \ddot q + \dot q^\\top B(q) \dot q + C(q) + sign(qdot) * D(q) = w,

    where w is a vector that satisfies the polyhedral constraint:

    .. math::
        F(q) w \\leq g(q).

    The functions `A, B, C, D` can represent respectively the
    inertial, Corriolis, gravitational and dry friction term for robot
    torque bound constraint.

    To evaluate the constraint on a geometric path `p(s)`, multiple
    calls to `inv_dyn` and `const_coeff` are made as follows:

    .. math::
        A(q) p'(s) \ddot s + [A(q) p''(s) + p'(s)^\\top B(q) p'(s)] \dot s^2 + C(q) + sign(p'(s)) * D(p(s)) = w,
        a(s) \ddot s + b(s) \dot s ^2 + c(s) = w.

    To evaluate the coefficients a(s), b(s), c(s), inv_dyn is called
    repeatedly with appropriate arguments.

    """

    def __init__(self, inv_dyn, cnst_F, cnst_g, dof, discretization_scheme=DiscretizationType.Interpolation):
        # type: ((np.array, np.array, np.array) -> np.ndarray)->None
        """Initialize the constraint.

        Parameters
        ----------
        inv_dyn: (array, array, array) -> array
            The "inverse dynamics" function that receives joint
            position, velocity and acceleration as inputs and ouputs
            the "joint torque". It is not necessary to supply each
            individual component functions such as gravitational,
            Coriolis, etc.

        cnst_F: array -> array
            Coefficient function. See notes for more details.
        cnst_g: array -> array
            Coefficient function. See notes for more details.
        dof: int, optional
            Dimension of joint position vectors. Required.

        """
        super(SecondOrderConstraint, self).__init__()
        self.set_discretization_type(discretization_scheme)
        self.inv_dyn = inv_dyn
        self.cnst_F = cnst_F
        self.cnst_g = cnst_g
        self.dof = dof
        self._format_string = "    Kind: Generalized Second-order constraint\n"
        self._format_string = "    Dimension:\n"
        F_ = cnst_F(np.zeros(dof))
        self._format_string += "        F in R^({:d}, {:d})\n".format(*F_.shape)

    def compute_constraint_params(self, path, gridpoints, scaling):
        assert path.get_dof() == self.get_dof(), ("Wrong dimension: constraint dof ({:d}) "
                                                  "not equal to path dof ({:d})".format(
                                                      self.get_dof(), path.get_dof()))
        v_zero = np.zeros(path.get_dof())
        p = path.eval(gridpoints / scaling)
        ps = path.evald(gridpoints / scaling) / scaling
        pss = path.evaldd(gridpoints / scaling) / scaling ** 2

        F = np.array(list(map(self.cnst_F, p)))
        g = np.array(list(map(self.cnst_g, p)))
        c = np.array(
            [self.inv_dyn(p_, v_zero, v_zero) for p_ in p]
        )
        a = np.array(
            [self.inv_dyn(p_, v_zero, ps_) for p_, ps_ in zip(p, ps)]
        ) - c
        b = np.array(
            [self.inv_dyn(p_, ps_, pss_) for p_, ps_, pss_ in zip(p, ps, pss)]
        ) - c

        if self.discretization_type == DiscretizationType.Collocation:
            return a, b, c, F, g, None, None
        elif self.discretization_type == DiscretizationType.Interpolation:
            return canlinear_colloc_to_interpolate(a, b, c, F, g, None, None, gridpoints)
        else:
            raise NotImplementedError("Other form of discretization not supported!")
