from ..algorithm import ParameterizationAlgorithm
from ...constants import LARGE, SMALL, TINY, INFTY, CVXPY_MAXX
from ...constraint import ConstraintType

import numpy as np
import logging
from copy import deepcopy
logger = logging.getLogger(__name__)


class ReachabilityAlgorithm(ParameterizationAlgorithm):
    """Base class for all Reachability Analysis-based parameterization algorithms.


    Parameters
    ----------
    constraint_list: list of Constraint
    path: Interpolator
    gridpoints: (N+1,)array, optional
    solver_wrapper: str, optional
        Name of solver to use. If leave to be None, will select the
        most suitable solver wrapper.
    scaling: float, optional
        Scale the problem instance as if the path's duration from [0,
        s_end] to [0, s_end * scaling]. Note that the path is actually
        kept unchanged; only the coefficients as evaluated is
        modified. This is to make it easier on the user who wish to
        implement an Interpolator class by herself (which she should
        do).

        The default value is 1.0, which means no scaling. Any positive
        float is perfectly valid. However, one should attempt to
        choose a scaling factor that leads to unit velocity
        approximately. Choose -1 for automatic scaling.

    Notes
    -----
    There are two patterns this class implements.

    1) Together with `SolverWrapper`, it uses a Strategy pattern to achieve different
       solver configuration.

    2) The class itself implements a `Template` pattern.

    RA-based algorithm uses a `SolverWrapper` for most, if not all, computations.
    During initialization, a solver wrapper is constructed from the given variables
    and is used afterward.

    In addition to a generic path parameterization algorithm, a RA-based algorithm
    implement additionally three methods:
    - compute_controllable_sets
    - compute_reachable_sets
    - compute_feasible_sets

    """
    def __init__(self, constraint_list, path, gridpoints=None, solver_wrapper=None, scaling=1):
        super(ReachabilityAlgorithm, self).__init__(constraint_list, path, gridpoints=gridpoints)
        # gridpoint check
        if gridpoints is None:
            gridpoints = np.linspace(0, path.get_duration(), 100)
        if path.get_path_interval()[0] != gridpoints[0]:
            logger.fatal("Manually supplied gridpoints does not start from 0.")
            raise ValueError("Bad manually supplied gridpoints.")
        if path.get_path_interval()[1] != gridpoints[-1]:
            logger.fatal("Manually supplied gridpoints have endpoint "
                         "different from input path duration.")
            raise ValueError("Bad manually supplied gridpoints.")
        self.gridpoints = np.array(gridpoints)  # Attr
        self._N = len(gridpoints) - 1  # Number of stages. Number of point is _N + 1
        for i in range(self._N):
            assert gridpoints[i + 1] > gridpoints[i]

        # Check for conic constraints
        has_conic = False
        for c in constraint_list:
            if c.get_constraint_type() == ConstraintType.CanonicalConic:
                has_conic = True

        # path scaling for numerical stability
        if scaling < 0:  # automatic scaling factor selection
            scaling = 1.0
        # NOTE: by scaling the gridpoints, we indicate to the lower
        # level solver wrapper that scaling is to be done. The solver
        # wrapper will simply use
        # scaling = self.gridpoints[-1] / path.duration
        self.gridpoints = self.gridpoints * scaling

        # Select solver wrapper automatically
        if solver_wrapper is None:
            logger.debug("Solver wrapper not supplied. Choose solver wrapper automatically!")
            if has_conic:
                solver_wrapper = "ecos"
            else:
                solver_wrapper = "qpOASES"
            logger.debug("Select solver {:}".format(solver_wrapper))
        else:
            if has_conic:
                assert solver_wrapper.lower() in ['cvxpy', 'ecos'], "Problem has conic constraints, solver {:} is not suitable".format(solver_wrapper)
            else:
                assert solver_wrapper.lower() in ['cvxpy', 'qpoases', 'ecos', 'hotqpoases', 'seidel'], "Solver {:} not found".format(solver_wrapper)

        if solver_wrapper.lower() == "cvxpy":
            from toppra.solverwrapper.cvxpy_solverwrapper import cvxpyWrapper
            self.solver_wrapper = cvxpyWrapper(self.constraints, self.path, self.gridpoints)
        elif solver_wrapper.lower() == "qpoases":
            from toppra.solverwrapper.qpoases_solverwrapper import qpOASESSolverWrapper
            self.solver_wrapper = qpOASESSolverWrapper(self.constraints, self.path, self.gridpoints)
        elif solver_wrapper.lower() == "hotqpoases":
            from toppra.solverwrapper.hot_qpoases_solverwrapper import hotqpOASESSolverWrapper
            self.solver_wrapper = hotqpOASESSolverWrapper(self.constraints, self.path, self.gridpoints)
        elif solver_wrapper.lower() == "ecos":
            from toppra.solverwrapper.ecos_solverwrapper import ecosWrapper
            self.solver_wrapper = ecosWrapper(self.constraints, self.path, self.gridpoints)
        elif solver_wrapper.lower() == "seidel":
            from toppra.solverwrapper.cy_seidel_solverwrapper import seidelWrapper
            self.solver_wrapper = seidelWrapper(self.constraints, self.path, self.gridpoints)
        else:
            raise NotImplementedError("Solver wrapper {:} not found!".format(solver_wrapper))

    def compute_feasible_sets(self):
        """Compute the sets of feasible squared velocities.

        Returns
        -------
        X: (N+1,2)array,
            X[i] contains the lower and upper bound of the feasible
            squared path velocity at s[i].  If there is no feasible
            state, X[i] equals (np.nan, np.nan).

        """
        logger.debug("Start computing the feasible sets")
        nV = self.solver_wrapper.get_no_vars()
        Hzero = np.zeros((nV, nV))
        g_lower = np.zeros(nV)
        g_lower[0] = 1e-9
        g_lower[1] = 1
        X = np.zeros((self._N + 1, 2))
        self.solver_wrapper.setup_solver()
        for i in range(self._N + 1):
            X[i, 0] = self.solver_wrapper.solve_stagewise_optim(
                i, Hzero, g_lower, -CVXPY_MAXX, CVXPY_MAXX, -CVXPY_MAXX, CVXPY_MAXX)[1]
            X[i, 1] = self.solver_wrapper.solve_stagewise_optim(
                i, Hzero, -g_lower, -CVXPY_MAXX, CVXPY_MAXX, -CVXPY_MAXX, CVXPY_MAXX)[1]
            if logger.getEffectiveLevel() == logging.DEBUG:
                logger.debug("X[{:d}]={:}".format(i, X[i]))
        self.solver_wrapper.close_solver()
        for i in range(self._N + 1):
            if X[i, 0] < 0:
                X[i, 0] = 0
        return X

    def compute_controllable_sets(self, sdmin, sdmax):
        """Compute the sets of controllable squared path velocities.

        Parameters
        ----------
        sdmin: float
            Lower bound on the final path velocity.
        sdmax: float
            Upper bound on the final path velocity.

        Returns
        -------
        K: (N+1,2)array
            K[i] contains the upper and lower bounds of the set of
            controllable squared velocities at position s[i].
        """
        assert sdmin <= sdmax and 0 <= sdmin
        K = np.zeros((self._N + 1, 2))
        K[self._N] = [sdmin ** 2, sdmax ** 2]
        logger.debug("Start computing the controllable sets")
        self.solver_wrapper.setup_solver()
        for i in range(self._N - 1, -1, -1):
            K[i] = self._one_step(i, K[i + 1])
            if K[i, 0] < 0:
                K[i, 0] = 0
            # check for potential numerical stability issues
            if K[i, 1] < 1e-4:
                logger.warn("Badly conditioned problem. Controllable sets are too small "
                            "K[{:d}] = {:}. ".format(i, K[i]))
                logger.warn("Consider set scaling to -1 when initiating TOPPRA for automatic"
                            " problem scaling.")
            elif K[i, 1] > 1e4:
                logger.warn("Badly conditioned problem. Controllable sets are too large "
                            "K[{:d}] = {:}".format(i, K[i]))
                logger.warn("Consider set scaling to -1 when initiating TOPPRA for automatic"
                            " problem scaling.")
            if np.isnan(K[i]).any():
                logger.warn("K[{:d}]={:}. Path not parametrizable.".format(i, K[i]))
                return K
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("[Compute controllable sets] K_{:d}={:}".format(i, K[i]))

        self.solver_wrapper.close_solver()
        return K

    def _one_step(self, i, K_next):
        """ Perform the one-step operation.

        Parameters
        ----------
        i: int
            Stage index.
        K_next: (2,)array
            Two ends of the set of controllable path velocities at stage (i+1).

        Returns
        -------
        res: (2,)array
            Set of controllable squared path velocities K[i].
        """
        res = np.zeros(2)
        if np.isnan(K_next).any() or i < 0 or i > self._N:
            res[:] = np.nan
            return res

        nV = self.solver_wrapper.get_no_vars()
        g_upper = np.zeros(nV)
        g_upper[0] = 1e-9
        g_upper[1] = - 1
        x_upper = self.solver_wrapper.solve_stagewise_optim(
            i, None, g_upper, np.nan, np.nan, K_next[0], K_next[1])[1]
        # import ipdb; ipdb.set_trace()
        x_lower = self.solver_wrapper.solve_stagewise_optim(
            i, None, - g_upper, np.nan, np.nan, K_next[0], K_next[1])[1]
        res[:] = [x_lower, x_upper]
        return res

    def compute_parameterization(self, sd_start, sd_end, return_data=False):
        """ Compute a path parameterization.

        If there is no valid parameterization, simply return None(s).

        Parameters
        ----------
        sd_start: float
            Starting path velocity. Must be positive.
        sd_end: float
            Goal path velocity. Must be positive.
        return_data: bool, optional
            If is True, also return matrix K which contains the controllable sets.

        Returns
        -------
        sdd_vec: (N,) array or None
            Path accelerations.
        sd_vec: (N+1,) array None
            Path velocities.
        v_vec: (N,) array or None
            Auxiliary variables.
        K: (N+1, 2) array
            Return the controllable set if `return_data` is True.
        """
        assert sd_end >= 0 and sd_start >= 0, "Path velocities must be positive"
        K = self.compute_controllable_sets(sd_end, sd_end)
        if np.isnan(K).any():
            logger.warn("The set of controllable velocities at the beginning is empty!")
            if return_data:
                return None, None, None, K
            else:
                return None, None, None

        x_start = sd_start ** 2
        if x_start + SMALL < K[0, 0] or K[0, 1] + SMALL < x_start:
            logger.warn("The initial velocity is not controllable. {:f} not in ({:f}, {:f})".format(
                x_start, K[0, 0], K[0, 1]
            ))
            if return_data:
                return None, None, None, K
            else:
                return None, None, None

        N = self.solver_wrapper.get_no_stages()
        deltas = self.solver_wrapper.get_deltas()
        xs = np.zeros(N + 1)
        xs[0] = x_start
        us = np.zeros(N)
        v_vec = np.zeros((N, self.solver_wrapper.get_no_vars() - 2))

        self.solver_wrapper.setup_solver()
        i = 0
        while i < self._N:
            optim_res = self._forward_step(i, xs[i], K[i + 1])
            if np.isnan(optim_res[0]):
                # NOTE: This case happens because the constraint
                # K[i + 1, 0] <= x[i] + 2D u[i] <= K[i + 1, 1]
                # become just slightly infeasible.  This happens more often
                # with ECOS, which is an interior point solver, than
                # with qpoases or seidel, both of which are active set
                # solvers. The strategy used to handle this case is in the
                # next line: simply reduce the current velocity by 0.1% or
                # a very small value (TINY) and choose the large value.
                xs[i] = max(xs[i] - TINY, 0.999 * xs[i])  # a slightly more aggressive reduction
                logger.warn(
                    "A numerical error occurs: the instance is controllable "
                    "but forward pass fails. Attempt to try again with x[i] "
                    "slightly reduced.\n"
                    "x[{:d}] reduced from {:.6f} to {:.6f}".format(i, xs[i] + SMALL, xs[i]))
            else:
                us[i] = optim_res[0]
                # The below function min( , max( ,)) ensure that the
                # state x_{i+1} is controllable.  While this is
                # ensured theoretically by the existence of the
                # controllable sets, numerical errors might violate
                # this condition.
                x_next = xs[i] + 2 * deltas[i] * us[i]
                x_next = max(x_next - TINY, 0.9999 * x_next)
                xs[i + 1] = min(K[i + 1, 1],
                                max(K[i + 1, 0], x_next))
                logger.debug("[Forward pass] u[{:d}] = {:f}, x[{:d}] = {:f}".format(i, us[i], i + 1, xs[i + 1]))
                v_vec[i] = optim_res[2:]
                i += 1
        self.solver_wrapper.close_solver()

        sd_vec = np.sqrt(xs)
        sdd_vec = np.copy(us)
        if return_data:
            return sdd_vec, sd_vec, v_vec, K
        else:
            return sdd_vec, sd_vec, v_vec
