from ..algorithm import ParameterizationAlgorithm
from ...solverwrapper import cvxpyWrapper
from ...constants import LARGE, SMALL

import numpy as np
import logging
logger = logging.getLogger(__name__)


class ReachabilityAlgorithm(ParameterizationAlgorithm):
    """ Base class for all Reachability Analysis-based parameterization algorithms.

    Parameters
    ----------
    constraint_list: list of Constraint
    path: Interpolator
    gridpoints: array, optional
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

    def __init__(self, constraint_list, path, gridpoints=None, solver_wrapper='cvxpy'):
        super(ReachabilityAlgorithm, self).__init__(constraint_list, path, gridpoints=gridpoints)
        if solver_wrapper == 'cvxpy':
            self.solver_wrapper = cvxpyWrapper(self.constraints, self.path, self.gridpoints)
        else:
            self.solver_wrapper = cvxpyWrapper(self.constraints, self.path, self.gridpoints)

    def compute_feasible_sets(self):
        """ Return the set of feasible squared velocities along the path.

        Returns
        -------
        X: array, or list containing None
            Shape (_N+1, 2). The tuple X[i] contains the lower and upper bound of the
            feasible squared path velocity at s[i].  If there is no feasible state,
            X[i] is the tuple (None, None).

        """
        nV = self.solver_wrapper.get_no_vars()
        Hzero = np.zeros((nV, nV))
        g_lower = np.zeros(nV)
        g_lower[1] = 1
        X_lower = map(lambda i: self.solver_wrapper.solve_stagewise_optim(
                i, Hzero, g_lower, -LARGE, LARGE, -LARGE, LARGE)[1], range(self._N + 1))
        X_upper = map(lambda i: self.solver_wrapper.solve_stagewise_optim(
            i, Hzero, - g_lower, -LARGE, LARGE, -LARGE, LARGE)[1], range(self._N + 1))
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
            Shape (_N+1,2). The tuple K[i] contains the upper and lower bounds of
            the set of controllable squared velocities at position s[i].

        """
        assert sdmin <= sdmax and 0 <= sdmin
        K = np.zeros((self._N + 1, 2))
        K[self._N] = [sdmin ** 2, sdmax ** 2]
        logger.debug("Start solving for controllable sets")
        for i in range(self._N - 1, -1, -1):
            if i % 1 == 0: logger.debug("[Solve Controllable] i={:d}".format(i))
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
        if None in K_next or i < 0 or i > self._N:
            return [None, None]

        nV = self.solver_wrapper.get_no_vars()
        g_upper = np.zeros(nV)
        g_upper[1] = - 1
        x_upper = self.solver_wrapper.solve_stagewise_optim(
            i, None, g_upper, None, None, K_next[0], K_next[1])[1]
        x_lower = self.solver_wrapper.solve_stagewise_optim(
            i, None, - g_upper, None, None, K_next[0], K_next[1])[1]
        return [x_lower, x_upper]

    def compute_parameterization(self, sd_start, sd_end):
        assert sd_end >= 0 and sd_start >= 0, "Path velocities must be positive"
        K = self.compute_controllable_sets(sd_end, sd_end)

        if None in K[0]:
            logger.warn("The set of controllable velocities at the beginning is empty!")
            return None, None, None

        x_start = sd_start ** 2
        if x_start + SMALL < K[0, 0] or K[0, 1] + SMALL < x_start:
            logger.warn("The initial velocity is not controllable.")
            return None, None, None

        N = self.solver_wrapper.get_no_stages()
        deltas = self.solver_wrapper.get_deltas()
        xs = np.zeros(N + 1)
        xs[0] = x_start
        us = np.zeros(N)
        v_vec = np.zeros((N, self.solver_wrapper.get_no_vars() - 2))

        for i in range(self._N):
            optim_res = self._forward_step(i, xs[i], K[i + 1])
            if optim_res is None:
                us[i] = None
                xs[i + 1] = None
                v_vec[i] = None
            else:
                us[i] = optim_res[0]
                xs[i + 1] = max(0, xs[i] + 2 * deltas[i] * us[i])
                v_vec[i] = optim_res[2:]
        sd_vec = np.sqrt(xs)
        sdd_vec = np.copy(us)
        return sdd_vec, sd_vec, v_vec

