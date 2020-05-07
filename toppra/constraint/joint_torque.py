from .linear_constraint import LinearConstraint, canlinear_colloc_to_interpolate
from ..constraint import DiscretizationType
import numpy as np


class JointTorqueConstraint(LinearConstraint):
    """Joint Torque Constraint.

    A joint torque constraint is given by

    .. math::
        A(q) \ddot q + \dot q^\\top B(q) \dot q + C(q) + D( \dot q )= w,

    where w is a vector that satisfies the polyhedral constraint:

    .. math::
        F(q) w \\leq g(q).

    Notice that `inv_dyn(q, qd, qdd) = w` and that `cnsf_coeffs(q) =
    F(q), g(q)`.

    To evaluate the constraint on a geometric path `p(s)`, multiple
    calls to `inv_dyn` and `const_coeff` are made. Specifically one
    can derive the second-order equation as follows

    .. math::
        A(q) p'(s) \ddot s + [A(q) p''(s) + p'(s)^\\top B(q) p'(s)] \dot s^2 + C(q) + D( \dot q ) = w,
        a(s) \ddot s + b(s) \dot s ^2 + c(s) = w

    To evaluate the coefficients a(s), b(s), c(s), inv_dyn is called
    repeatedly with appropriate arguments.

    Parameters
    ----------

    inv_dyn: (array, array, array) -> array
        The "inverse dynamics" function that receives joint position, velocity and
        acceleration as inputs and ouputs the "joint torque". See notes for more
        details.

    tau_lim: array
        Shape (dof, 2). The lower and upper torque bounds of the
        j-th joint are tau_lim[j, 0] and tau_lim[j, 1] respectively.

    fs_coef: array
        Shape (dof). The coefficients of dry friction of the
        joints.

    discretization_scheme: :class:`.DiscretizationType`
        Can be either Collocation (0) or Interpolation
        (1). Interpolation gives more accurate results with slightly
        higher computational cost.

    """

    def __init__(
        self,
        inv_dyn,
        tau_lim,
        fs_coef,
        discretization_scheme=DiscretizationType.Collocation,
    ):
        super(JointTorqueConstraint, self).__init__()
        self.inv_dyn = inv_dyn
        self.tau_lim = np.array(tau_lim, dtype=float)
        self.fs_coef = np.array(fs_coef)
        self.dof = self.tau_lim.shape[0]
        self.set_discretization_type(discretization_scheme)
        assert self.tau_lim.shape[1] == 2, "Wrong input shape."
        self._format_string = "    Torque limit: \n"
        for i in range(self.tau_lim.shape[0]):
            self._format_string += (
                "      J{:d}: {:}".format(i + 1, self.tau_lim[i]) + "\n"
            )
        self.identical = True

    def compute_constraint_params(self, path, gridpoints):
        if path.dof != self.get_dof():
            raise ValueError(
                "Wrong dimension: constraint dof ({:d}) not equal to path dof ({:d})".format(
                    self.get_dof(), path.dof
                )
            )
        v_zero = np.zeros(path.dof)
        p = path.eval(gridpoints)
        ps = path.evald(gridpoints)
        pss = path.evaldd(gridpoints)
        N = gridpoints.shape[0] - 1
        dof = path.dof
        I_dof = np.eye(dof)
        F = np.zeros((dof * 2, dof))
        g = np.zeros(dof * 2)
        g[0:dof] = self.tau_lim[:, 1]
        g[dof:] = -self.tau_lim[:, 0]
        F[0:dof, :] = I_dof
        F[dof:, :] = -I_dof

        c = np.array([self.inv_dyn(p_, v_zero, v_zero) for p_ in p])
        a = np.array([self.inv_dyn(p_, v_zero, ps_) for p_, ps_ in zip(p, ps)]) - c
        b = (
            np.array([self.inv_dyn(p_, ps_, pss_) for p_, ps_, pss_ in zip(p, ps, pss)])
            - c
        )

        # dry friction
        for i in range(0, dof):
            c[:, i] += self.fs_coef[i] * np.sign(ps[:, i])

        if self.discretization_type == DiscretizationType.Collocation:
            return a, b, c, F, g, None, None
        elif self.discretization_type == DiscretizationType.Interpolation:
            return canlinear_colloc_to_interpolate(
                a, b, c, F, g, None, None, gridpoints, identical=True
            )
        else:
            raise NotImplementedError("Other form of discretization not supported!")
