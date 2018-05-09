from .canonical_linear import CanonicalLinearConstraint, canlinear_colloc_to_interpolate
from .constraint import DiscretizationType
import numpy as np


class CanonicalLinearSecondOrderConstraint(CanonicalLinearConstraint):
    """ A class to represent Canonical Linear Generalized Second-order constraints.

    Parameters
    ----------
    inv_dyn: (array, array, array) -> array
        The "inverse dynamics" function that receives joint position, velocity and
        acceleration as inputs and ouputs the "joint torque". See notes for more
        details.
    cnst_coeffs: (array) -> array, array
        The coefficient functions of the constraints. See notes for more details.

    Notes
    -----
    A constraint of this kind can be represented by the following formula

    .. math::
        A(q) \ddot q + \dot q^\\top B(q) \dot q + C(q) = w,

    where w is a vector that satisfies the polyhedral constraint

    .. math::
        F(q) w \\leq g(q).

    To evaluate the constraint parameters, multiple calls to inv_dyn, cnst_F and cnst_g
    are made. Specifically one can write the second-order equation as follows

    .. math::
        A(q) p'(s) \ddot s + [A(q) p''(s) + p'(s)^\\top B(q) p'(s)] + C(q) = w,

    To evaluate the coefficients a(s), b(s), c(s), inv_dyn is called repeatedly with
    appropriate arguments.
    """

    def __init__(self, inv_dyn, cnst_F, cnst_g, dof=None, discretization_scheme=DiscretizationType.Collocation):
        super(CanonicalLinearSecondOrderConstraint, self).__init__()
        self.discretization_type = discretization_scheme
        self.inv_dyn = inv_dyn
        self.cnst_F = cnst_F
        self.cnst_g = cnst_g
        self._format_string = "    Kind: Generalized Second-order constraint\n"
        self._format_string = "    Dimension:\n"
        if dof is not None:
            z_ = np.zeros(dof)
            F_ = cnst_F(z_)
        self._format_string += "        F in R^({:d}, {:d})\n".format(*F_.shape)
        self.discretization_type = discretization_scheme

    def compute_constraint_params(self, path, gridpoints):
        v_zero = np.zeros(path.get_dof())
        p = path.eval(gridpoints)
        ps = path.evald(gridpoints)
        pss = path.evaldd(gridpoints)

        F = np.array(map(self.cnst_F, p))
        g = np.array(map(self.cnst_g, p))
        c = np.array(
            map(lambda p_: self.inv_dyn(p_, v_zero, v_zero), p)
        )
        a = np.array(
            map(lambda (p_, ps_): self.inv_dyn(p_, v_zero, ps_), zip(p, ps))
        ) - c
        b = np.array(
            map(lambda (p_, ps_, pss_): self.inv_dyn(p_, ps_, pss_), zip(p, ps, pss))
        ) - c

        if self.discretization_type == DiscretizationType.Collocation:
            return a, b, c, F, g, None, None
        elif self.discretization_type == DiscretizationType.Interpolation:
            return canlinear_colloc_to_interpolate(a, b, c, F, g, None, None, gridpoints)
        else:
            raise NotImplementedError, "Other form of discretization not supported!"

