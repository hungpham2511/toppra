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

        if x_min is not None:
            cvxpy_constraints.append(x_min <= x)
        if x_max is not None:
            cvxpy_constraints.append(x <= x_max)

        if i < self.N:
            delta = self.get_deltas()[i]
            if x_next_min is not None:
                cvxpy_constraints.append(x_next_min <= x + 2 * delta * u)
            if x_next_max is not None:
                cvxpy_constraints.append(x + 2 * delta * u <= x_next_max)

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

        if H is None:
            H = np.zeros((self.get_no_vars(), self.get_no_vars()))

        objective = cvxpy.Minimize(0.5 * cvxpy.quad_form(ux, H) + g * ux)

        problem = cvxpy.Problem(objective, constraints=cvxpy_constraints)
        problem.solve(solver='MOSEK')
        if problem.status == 'optimal':
            return np.array(ux.value).flatten()
        else:
            return [None] * self.get_no_vars()

