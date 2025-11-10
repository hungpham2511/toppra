"""This module implements the general Second-Order constraints."""
import logging
import numpy as np

from .linear_constraint import LinearConstraint, canlinear_colloc_to_interpolate
from .constraint import DiscretizationType

logger = logging.getLogger(__name__)


class SecondOrderConstraint(LinearConstraint):
    """This class implements the linear Second-Order constraint.

    Conventionally, a :class:`SecondOrderConstraint` is given by the
    following formula:

    .. math::
        A(\mathbf{q}) \ddot {\mathbf{q}} + \dot
        {\mathbf{q}}^\\top B(\mathbf{q}) \dot {\mathbf{q}} + C(\mathbf{q}) = w,

    where w is a vector that satisfies the polyhedral constraint:

    .. math::
        F(\mathbf{q}) w \\leq g(\mathbf{q}).

    Take the example of a robot torque bound, the functions :math:`A,
    B, C` represent respectively the inertial, Corriolis and
    gravitational terms of the robot's rigid body dynamics.

    We can evaluate a constraint on a given geometric path
    :math:`\mathbf{p}(s)` using the following equations, which are
    obtained by direct substitution:

    .. math::

        A(\mathbf{q}) \mathbf{p}'(s) \ddot s + [A(\mathbf{q}) \mathbf{p}''(s) + \mathbf{p}'(s)^\\top B(\mathbf{q})
        \mathbf{p}'(s)] \dot s^2 + C(\mathbf{q}) = w, \\\\
        \mathbf{a}(s) \ddot s + \mathbf{b}(s) \dot s ^2 + \mathbf{c}(s) = w.

    where :math:`\mathbf{p}', \mathbf{p}''` denote respectively the
    first and second derivatives of the path. It is important to
    understand that the vector functions :math:`\mathbf a, \mathbf b,
    \mathbf c` are what `toppra` needs to solve for path
    parametrizations.

    To evaluate these coefficients :math:`\mathbf a(s), \mathbf b(s),
    \mathbf c(s)`, fortunately, it is not necessary to have the
    functions :math:`A, B, C` explicitly. Rather, it is only required
    to have the sum of the these 3 functions--the so-called inverse
    dynamic function:

    .. math::
        \mathrm{inverse\_dyn}(\mathbf q, \dot{\mathbf q}, \ddot{\mathbf q}) :=
        A(\mathbf{q}) \ddot {\mathbf{q}} + \dot {\mathbf{q}}^\\top B(\mathbf{q}) \dot {\mathbf{q}} + C(\mathbf{q})

    In some cases, one might have terms that depends purely on the
    path:

    .. math::
        \mathbf{a}(s) \ddot s + \mathbf{b}(s) \dot s ^2 + \mathbf{c}(s) + \mathcal{C}(\mathbf p, s)= w.

    an example is the joint friction. This term is referred to as
    `custom_term` in the initializing arguments of
    :class:`SecondOrderConstraint`.

    It is interesting to note that we can actually use a more general
    form of the above equations, hence covering a wider class of
    constraints. In particular, one can replace :math:`A(\mathbf{q}),
    B(\mathbf{q}), C(\mathbf{q}), F(\mathbf{q}), g(\mathbf{q})` with
    :math:`A(\mathbf{q}, s), B(\mathbf{q}, s), C(\mathbf{q}, s),
    F(\mathbf{q}, s), g(\mathbf{q}, s)`. This form, however, is not
    implemented in `toppra`.

    """

    def __init__(self, inv_dyn, constraint_F, constraint_g, dof, custom_term=None, discretization_scheme=1):
        """Initialize the constraint.

        Parameters
        ----------
        inv_dyn: (np.ndarray, np.ndarray, np.ndarray) -> np.ndarray
            The "inverse dynamics" function that receives joint
            position, velocity, acceleration and path position as inputs and ouputs
            the constrained vector :math:`\mathbf w`. See above for more details.
        constraint_F: (np.ndarray) -> np.ndarray
            The constraint coefficient function :math:`\mathbf
            F(\mathbf q, s)`. See above for more details.
        constraint_g: (np.ndarray) -> np.ndarray
            The constraint coefficient function :math:`\mathbf
            g(\mathbf q, s)`. See above for more details.
        dof: int
            The dimension of the joint position.
        custom_term: (:class:`Interpolator`, float) -> np.ndarray
            This function receives as input a geometric path and a
            float path position, then returns an additive term. See
            the above note for more details.

        discretization_scheme: DiscretizationType
            Type of discretization.

        """
        super(SecondOrderConstraint, self).__init__()
        self.set_discretization_type(discretization_scheme)
        self.inv_dyn = inv_dyn
        self.constraint_F = constraint_F
        self.constraint_g = constraint_g
        self.dof = dof
        self.custom_term = custom_term
        self._format_string = "    Kind: Generalized Second-order constraint\n"
        self._format_string = "    Dimension:\n"
        self._format_string += "        F in R^({:d}, {:d})\n".format(
            *constraint_F(np.zeros(dof)).shape)

    @classmethod
    def joint_torque_constraint(cls, inv_dyn, taulim, joint_friction,
                                **kwargs):
        """Initialize a Joint Torque constraint.

        Parameters
        ----------
        inv_dyn: (np.ndarray, np.ndarray, np.ndarray) -> np.ndarray
            Inverse dynamic function of the robot.
        taulim: np.ndarray
            Shape (N, 2). The i-th element contains the minimum and
            maximum joint torque limits respectively.
        joint_friction: np.ndarray
            Shape (N,). The i-th element contains the dry friction (torque)
            in the i-th joint.

        """
        dof = np.shape(taulim)[0]
        stacked_eyes = np.vstack((np.eye(dof), -np.eye(dof)))
        g_aug = np.zeros(2 * dof)
        g_aug[:dof] = taulim[:, 1]
        g_aug[dof:] = -taulim[:, 0]
        constraint_F = lambda _: stacked_eyes
        constraint_g = lambda _: g_aug
        custom_term = lambda path, s: np.sign(path(s, 1)) * joint_friction
        return SecondOrderConstraint(inv_dyn, constraint_F, constraint_g, dof, custom_term,
                                     **kwargs)

    def compute_constraint_params(self, path, gridpoints):
        if path.dof != self.dof:
            raise ValueError(
                "Wrong dimension: constraint dof ({:d}) not equal to path dof ({:d})"
                .format(self.dof, path.dof))
        v_zero = np.zeros(path.dof)
        p_vec = path(gridpoints)
        ps_vec = path(gridpoints, 1)
        pss_vec = path(gridpoints, 2)

        F_vec = np.array(list(map(self.constraint_F, p_vec)))
        g_vec = np.array(list(map(self.constraint_g, p_vec)))
        c_vec = np.array([self.inv_dyn(_p, v_zero, v_zero) for _p in p_vec])

        a_vec = np.array(
            [self.inv_dyn(_p, v_zero, _ps)
             for _p, _ps in zip(p_vec, ps_vec)]) - c_vec
        b_vec = np.array([
            self.inv_dyn(_p, _ps, pss_)
            for _p, _ps, pss_ in zip(p_vec, ps_vec, pss_vec)
        ]) - c_vec
        if self.custom_term is not None:
            for i, _ in enumerate(gridpoints):
                c_vec[i] = c_vec[i] + self.custom_term(path, gridpoints[i])

        if self.discretization_type == DiscretizationType.Collocation:
            return a_vec, b_vec, c_vec, F_vec, g_vec, None, None
        if self.discretization_type == DiscretizationType.Interpolation:
            return canlinear_colloc_to_interpolate(a_vec, b_vec, c_vec, F_vec,
                                                   g_vec, None, None,
                                                   gridpoints)
        raise NotImplementedError("Other form of discretization not supported!")
