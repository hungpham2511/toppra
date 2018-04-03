from .solverwrapper import SolverWrapper
import cvxpy
import numpy as np
from ..constraint import ConstraintType


class cvxpyWrapper(SolverWrapper):
    """ A solver wrapper using `cvxpy`.

    Parameters
    ----------
    constraint_list: list of :class:`.Constraint`
        The constraints the robot is subjected to.
    path: :class:`.Interpolator`
        The geometric path.
    path_discretization: array
        The discretized path positions.
    """

    def __init__(self, constraint_list, path, path_discretization):
        super(cvxpyWrapper, self).__init__(constraint_list, path, path_discretization)

        # Currently only support Canonical Linear Constraint
        for constraint in constraint_list:
            if constraint.get_constraint_type() != ConstraintType.CanonicalLinear:
                raise NotImplementedError

    def solve_stagewise_optim(self, i, H, g, x_min, x_max, x_next_min, x_next_max):
        assert i <= self.N and 0 <= i

        ux = cvxpy.Variable(2)
        u = ux[0]
        x = ux[1]
        cvxpy_constraints = []
        for j in range(len(self.constraints)):
            a, b, c, F, h, ubound, xbound = self.params[j]

            # Case 1
            if a is not None:
                v = a[i] * u + b[i] * x + c[i]
                cvxpy_constraints.append(F[i] * v <= h[i])

            if ubound is not None:
                cvxpy_constraints.append(ubound[i, 0] <= u)
                cvxpy_constraints.append(u <= ubound[i, 1])

            if xbound is not None:
                cvxpy_constraints.append(xbound[i, 0] <= x)
                cvxpy_constraints.append(x <= xbound[i, 1])

        objective = cvxpy.Minimize(0.5 * cvxpy.quad_form(ux, H) + g * ux)

        problem = cvxpy.Problem(objective, constraints=cvxpy_constraints)
        optimal_value = problem.solve()
        if isinstance(optimal_value, float):
            return np.array(ux.value)
        else:
            return None


