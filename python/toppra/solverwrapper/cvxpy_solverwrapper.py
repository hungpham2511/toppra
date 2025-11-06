from .solverwrapper import SolverWrapper
import logging
import numpy as np
from ..constraint import ConstraintType
from ..constants import CVXPY_MAXX, CVXPY_MAXU


logger = logging.getLogger(__name__)
try:
    import cvxpy

    FOUND_CVXPY = True
except ImportError:
    logger.info("CVXPY installation not found.")
    FOUND_CVXPY = False
try:
    import mosek

    FOUND_MOSEK = True
except ImportError:
    logger.info("Mosek installation not found!")
    FOUND_MOSEK = False


class cvxpyWrapper(SolverWrapper):
    """A solver wrapper using `cvxpy`.

    NOTE: the two constants CVXPY_MAXX and CVXPY_MAXU is used to
    guarantee that the solution is not too large, in which case cvxpy
    can't handle very well.

    `cvxpyWrapper` should not be used in production due to robustness
    issue.

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
        valid_types = [ConstraintType.CanonicalLinear, ConstraintType.CanonicalConic]
        # Currently only support Canonical Linear Constraint
        for constraint in constraint_list:
            if constraint.get_constraint_type() not in valid_types:
                raise NotImplementedError

    def solve_stagewise_optim(self, i, H, g, x_min, x_max, x_next_min, x_next_max):
        assert i <= self.N and 0 <= i

        ux = cvxpy.Variable(2)
        u = ux[0]
        x = ux[1]
        cvxpy_constraints = [-CVXPY_MAXU <= u, u <= CVXPY_MAXU, 0 <= x, x <= CVXPY_MAXX]

        if not np.isnan(x_min):
            cvxpy_constraints.append(x_min <= x)
        if not np.isnan(x_max):
            cvxpy_constraints.append(x <= x_max)

        if i < self.N:
            delta = self.get_deltas()[i]
            if not np.isnan(x_next_min):
                cvxpy_constraints.append(x_next_min <= x + 2 * delta * u)
            if not np.isnan(x_next_max):
                cvxpy_constraints.append(x + 2 * delta * u <= x_next_max)

        for k, constraint in enumerate(self.constraints):
            if constraint.get_constraint_type() == ConstraintType.CanonicalLinear:
                a, b, c, F, h, ubound, xbound = self.params[k]

                if a is not None:
                    v = a[i] * u + b[i] * x + c[i]
                    if constraint.identical:
                        cvxpy_constraints.append(F * v <= h)
                    else:
                        cvxpy_constraints.append(F[i] * v <= h[i])

                # ecos (via cvxpy in this class) is very bad at
                # handling badly scaled problems. Problems with very
                # large bound. The below max(), min() operators is a
                # workaround to get pass this issue.
                if ubound is not None:
                    cvxpy_constraints.append(max(-CVXPY_MAXU, ubound[i, 0]) <= u)
                    cvxpy_constraints.append(u <= min(CVXPY_MAXU, ubound[i, 1]))

                if xbound is not None:
                    cvxpy_constraints.append(xbound[i, 0] <= x)
                    cvxpy_constraints.append(x <= min(CVXPY_MAXX, xbound[i, 1]))

            elif constraint.get_constraint_type() == ConstraintType.CanonicalConic:
                a, b, c, P, ubound, xbound = self.params[k]

                if a is not None:
                    d = a.shape[1]
                    for j in range(d):
                        cvxpy_constraints.append(
                            a[i, j] * u
                            + b[i, j] * x
                            + c[i, j]
                            + cvxpy.norm(P[i, j].T[:, :2] * ux + P[i, j].T[:, 2])
                            <= 0
                        )

                if ubound is not None:
                    cvxpy_constraints.append(max(-CVXPY_MAXU, ubound[i, 0]) <= u)
                    cvxpy_constraints.append(u <= min(CVXPY_MAXU, ubound[i, 1]))

                if xbound is not None:
                    cvxpy_constraints.append(xbound[i, 0] <= x)
                    cvxpy_constraints.append(x <= min(CVXPY_MAXX, xbound[i, 1]))

        if H is None:
            H = np.zeros((self.get_no_vars(), self.get_no_vars()))

        objective = cvxpy.Minimize(0.5 * cvxpy.quad_form(ux, H) + g * ux)
        problem = cvxpy.Problem(objective, constraints=cvxpy_constraints)
        try:
            problem.solve(verbose=False)
        except cvxpy.SolverError:
            # solve fail
            pass
        if (
            problem.status == cvxpy.OPTIMAL
            or problem.status == cvxpy.OPTIMAL_INACCURATE
        ):
            return np.array(ux.value).flatten()
        else:
            res = np.empty(self.get_no_vars())
            res[:] = np.nan
            return res
