from ..algorithm import ParameterizationAlgorithm
from ...solverwrapper import cvxpyWrapper
from ...constants import LARGE
import numpy as np


class ReachabilityAlgorithm(ParameterizationAlgorithm):
    """ Base class for all Reachability Analysis-based parameterization algorithms.

    Parameters
    ----------
    constraint_list:
    path:
    path_discretization: array, optional
    solver_wrapper: str, optional
        Name of the solver to use.

    RA-based algorithm uses a `SolverWrapper` for most, if not all, computations.
    During initialization, a solver wrapper is constructed from the given variables
    and is used afterward.

    In contrast to a generic path parameterization algorithm, a RA-based algorithm
    implement additionally three methods:
    - compute_controllable_sets
    - compute_reachable_sets
    - compute_feasible_sets


    """

    def __init__(self, constraint_list, path, path_discretization=None, solver_wrapper='cvxpy'):
        super(ReachabilityAlgorithm, self).__init__(constraint_list, path, path_discretization=path_discretization)
        if solver_wrapper=='cvxpy':
            self.solver_wrapper = cvxpyWrapper(self.constraints, self.path, self.path_discretization)
        else:
            self.solver_wrapper = cvxpyWrapper(self.constraints, self.path, self.path_discretization)

    def compute_feasible_sets(self):
        """ Return the set of feasible squared velocities along the path.

        Returns
        -------
        X: array, or list containing None
            Shape (N+1, 2). The tuple X[i] contains the lower and upper bound of the
            feasible squared path velocity at s[i].  If there is no feasible state,
            X[i] is the tuple (None, None).

        """
        nV = self.solver_wrapper.get_no_vars()
        Hzero = np.zeros((nV, nV))
        g_lower = np.zeros(nV)
        g_lower[1] = 1
        X_lower = map(lambda i: self.solver_wrapper.solve_stagewise_optim(
                i, Hzero, g_lower, -LARGE, LARGE, -LARGE, LARGE)[1], range(self.N + 1))
        X_upper = map(lambda i: self.solver_wrapper.solve_stagewise_optim(
            i, Hzero, - g_lower, -LARGE, LARGE, -LARGE, LARGE)[1], range(self.N + 1))
        X = np.array((X_lower, X_upper)).T
        return X

    def compute_controllable_sets(self, sdmin, sdmax):
        """ Compute the set of controllable squared velocities.

        Parameters
        ----------
        sdmin: float
            Lower bound on the final path velocity.
        sdmax: float
            Upper bound on the final path velocity.

        Returns
        -------
        K: array
            Shape (N+1,2). The tuple K[i] contains the upper and lower bounds of
            the set of controllable squared velocities at position s[i].

        """
        assert sdmin <= sdmax and 0 <= sdmin
        K = np.zeros((self.N + 1, 2))
        K[self.N] = [sdmin ** 2, sdmax ** 2]
        for i in range(self.N - 1, -1, -1):
            K[i] = self._one_step(i, K[i + 1])
        return K

    def _one_step(self, i, K_next):
        """

        Parameters
        ----------
        i: int
        K_next: list of float or None

        Returns
        -------
        K: list of 2 float or None
        """
        if None in K_next or i < 0 or i > self.N:
            return [None, None]

        nV = self.solver_wrapper.get_no_vars()
        Hzero = np.zeros((nV, nV))
        g_upper = np.zeros(nV)
        g_upper[1] = - 1
        # g_upper[0] = - self.solver_wrapper.get_deltas()[i] * 2
        x_upper = self.solver_wrapper.solve_stagewise_optim(
            i, Hzero, g_upper, -LARGE, LARGE, K_next[0], K_next[1])[1]
        x_lower = self.solver_wrapper.solve_stagewise_optim(
            i, Hzero, - g_upper, -LARGE, LARGE, K_next[0], K_next[1])[1]
        return [x_lower, x_upper]

