from ..algorithm import ParameterizationAlgorithm, ParameterizationReturnCode
from ...constants import LARGE, SMALL, TINY, INFTY, CVXPY_MAXX, MAX_TRIES
from ...constraint import ConstraintType
import toppra.solverwrapper
import toppra.exceptions as exceptions
import toppra.interpolator as interpolator

import numpy as np
import logging

logger = logging.getLogger(__name__)


class ReachabilityAlgorithm(ParameterizationAlgorithm):
    """Base class for Reachability Analysis-based path parameterization algorithms.

    Parameters
    ----------
    constraint_list: List[:class:`~toppra.constraint.Constraint`]
        List of constraints on the robot dynamics.
    path: Interpolator
    gridpoints: np.ndarray, optional
        Shape (N+1,). Gridpoints for discretization of the path position.
    solver_wrapper: str, optional
        Name of solver to use. If is None, select the most suitable wrapper.
    parametrizer: str, optional
        Name of the output parametrizer to use.

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

    def __init__(
            self, constraint_list, path, gridpoints=None, solver_wrapper=None, parametrizer=None, **kwargs
    ):
        super(ReachabilityAlgorithm, self).__init__(
            constraint_list, path, gridpoints=gridpoints, parametrizer=parametrizer, **kwargs
        )

        # Check for conic constraints
        has_conic = False
        for c in constraint_list:
            if c.get_constraint_type() == ConstraintType.CanonicalConic:
                has_conic = True

        # Select solver wrapper automatically
        available_solvers = toppra.solverwrapper.available_solvers(output_msg=False)
        if solver_wrapper is None:
            logger.info(
                "Solver wrapper not supplied. Choose solver wrapper automatically!"
            )
            if has_conic:
                if not available_solvers["ecos"]:
                    raise exceptions.ToppraError("Solverwrapper not available.")
                solver_wrapper = "ecos"
            else:
                valid_solver = [solver for solver, avail in available_solvers if avail]
                solver_wrapper = valid_solver[0]
            logger.info("Select solver {:}".format(solver_wrapper))

        # Check solver-wrapper suitability
        if has_conic:
            assert solver_wrapper.lower() in [
                "cvxpy",
                "ecos",
            ], "Problem has conic constraints, solver {:} is not suitable".format(
                solver_wrapper
            )
        else:
            assert solver_wrapper.lower() in [
                "cvxpy",
                "qpoases",
                "ecos",
                "hotqpoases",
                "seidel",
            ], "Solver {:} not found".format(solver_wrapper)

        if solver_wrapper.lower() == "cvxpy":
            from toppra.solverwrapper.cvxpy_solverwrapper import cvxpyWrapper

            self.solver_wrapper = cvxpyWrapper(
                self.constraints, self.path, self.gridpoints
            )
        elif solver_wrapper.lower() == "qpoases":
            from toppra.solverwrapper.qpoases_solverwrapper import qpOASESSolverWrapper

            self.solver_wrapper = qpOASESSolverWrapper(
                self.constraints, self.path, self.gridpoints
            )
        elif solver_wrapper.lower() == "hotqpoases":
            from toppra.solverwrapper.hot_qpoases_solverwrapper import (
                hotqpOASESSolverWrapper,
            )

            self.solver_wrapper = hotqpOASESSolverWrapper(
                self.constraints, self.path, self.gridpoints
            )
        elif solver_wrapper.lower() == "ecos":
            from toppra.solverwrapper.ecos_solverwrapper import ecosWrapper

            self.solver_wrapper = ecosWrapper(
                self.constraints, self.path, self.gridpoints
            )
        elif solver_wrapper.lower() == "seidel":
            from toppra.solverwrapper.cy_seidel_solverwrapper import seidelWrapper

            self.solver_wrapper = seidelWrapper(
                self.constraints, self.path, self.gridpoints
            )
        else:
            raise NotImplementedError(
                "Solver wrapper {:} not found!".format(solver_wrapper)
            )

    def compute_feasible_sets(self):
        """Compute the sets of feasible squared velocities.

        Returns
        -------
        X: array
            Shape (N+1,2). X[i] contains the lower and upper bound of
            the feasible squared path velocity at s[i].  If there is
            no feasible state, X[i] equals (np.nan, np.nan).

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
                i, Hzero, g_lower, -CVXPY_MAXX, CVXPY_MAXX, -CVXPY_MAXX, CVXPY_MAXX
            )[1]
            X[i, 1] = self.solver_wrapper.solve_stagewise_optim(
                i, Hzero, -g_lower, -CVXPY_MAXX, CVXPY_MAXX, -CVXPY_MAXX, CVXPY_MAXX
            )[1]
            if logger.getEffectiveLevel() == logging.DEBUG:
                logger.debug("X[{:d}]={:}".format(i, X[i]))
        self.solver_wrapper.close_solver()
        for i in range(self._N + 1):
            if X[i, 0] < 0:
                X[i, 0] = 0
        self._problem_data.X = X
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
        K : array
            Shape (N+1, 2). Element K[i] contains the squared upper
            and lower controllable velocities at position s[i].

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
            if np.isnan(K[i]).any():
                logger.warning(
                    "A numerical error occurs: The controllable set at step "
                    "[{:d} / {:d}] can't be computed.".format(i, self._N + 1)
                )
                return K
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("[Compute controllable sets] K_{:d}={:}".format(i, K[i]))

        self.solver_wrapper.close_solver()
        return K

    def _one_step(self, i, K_next):
        """Perform the one-step operation.

        Parameters
        ----------
        i: int
            Stage index.
        K_next: array
            Shape(2,).Two ends of the set of controllable path
            velocities at stage (i+1).

        Returns
        -------
        res: array
            Shape (2,). Set of controllable squared path velocities
            K[i].

        """
        res = np.zeros(2)
        if np.isnan(K_next).any() or i < 0 or i > self._N:
            res[:] = np.nan
            return res

        nV = self.solver_wrapper.get_no_vars()
        g_upper = np.zeros(nV)
        g_upper[0] = 1e-9
        g_upper[1] = -1
        x_upper = self.solver_wrapper.solve_stagewise_optim(
            i, None, g_upper, np.nan, np.nan, K_next[0], K_next[1]
        )[1]
        x_lower = self.solver_wrapper.solve_stagewise_optim(
            i, None, -g_upper, np.nan, np.nan, K_next[0], K_next[1]
        )[1]
        res[:] = [x_lower, x_upper]
        return res

    def compute_parameterization(self, sd_start, sd_end, return_data=False):
        """Compute a path parameterization.

        If fail, whether because there is no valid parameterization or
        because of numerical error, the arrays returns should contain
        np.nan.

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
        sdd_vec: array
            Shape (N,). Path accelerations. Double array. Will contain
            nan(s) if failed.

        sd_vec: array
            Shape (N+1,). Path velocities. Double array. Will contain nan(s) if failed.

        v_vec: array or None
            Shape (N,). Auxiliary variables.

        K: array
            Shape (N+1, 2). Return the controllable set if
            `return_data` is True.
        """
        if sd_end < 0 or sd_start < 0:
            raise exceptions.BadInputVelocities(
                "Negative path velocities: path velocities must be positive: (%s, %s)"
                % (sd_start, sd_end)
            )
        K = self.compute_controllable_sets(sd_end, sd_end)
        if np.isnan(K).any():
            logger.warning(
                "An error occurred when computing controllable velocities. "
                "The path is not controllable, or is badly conditioned."
            )
            self._problem_data.return_code = ParameterizationReturnCode.FailUncontrollable
            if return_data:
                return None, None, None, K
            else:
                return None, None, None
        self._problem_data.K = K

        x_start = sd_start ** 2
        if x_start + SMALL < K[0, 0] or K[0, 1] + SMALL < x_start:
            logger.warning(
                "The initial velocity is not controllable. {:f} not in ({:f}, {:f})".format(
                    x_start, K[0, 0], K[0, 1]
                )
            )
            self._problem_data.return_code = ParameterizationReturnCode.FailUncontrollable
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

        tries = 0
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
                if tries < MAX_TRIES:
                    xs[i] = max(
                        xs[i] - TINY, 0.999 * xs[i]
                    )  # a slightly more aggressive reduction
                    tries += 1
                    logger.warning(
                        "A numerical error occurs: the instance is controllable "
                        "but forward pass fails. Attempt to try again with x[i] "
                        "slightly reduced.\n"
                        "x[{:d}] reduced from {:.6f} to {:.6f}".format(
                            i, xs[i] + SMALL, xs[i]
                        )
                    )
                else:
                    logger.warning(
                        "Number of trials (to reduce xs[i]) reaches limits. "
                        "Compute parametrization fails!"
                    )
                    xs[i + 1 :] = np.nan
                    break
            else:
                tries = 0
                us[i] = optim_res[0]
                # The below function min( , max( ,)) ensure that the
                # state x_{i+1} is controllable.  While this is
                # ensured theoretically by the existence of the
                # controllable sets, numerical errors might violate
                # this condition.
                x_next = xs[i] + 2 * deltas[i] * us[i]
                x_next = max(x_next - TINY, 0.9999 * x_next)
                xs[i + 1] = min(K[i + 1, 1], max(K[i + 1, 0], x_next))
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "[Forward pass] u[{:d}] = {:f}, x[{:d}] = {:f}".format(
                            i, us[i], i + 1, xs[i + 1]
                        )
                    )
                v_vec[i] = optim_res[2:]
                i += 1
        self.solver_wrapper.close_solver()

        sd_vec = np.sqrt(xs)
        sdd_vec = np.copy(us)
        self._problem_data.sd_vec = sd_vec
        self._problem_data.sdd_vec = sdd_vec
        if np.isnan(sd_vec).any():
            self._problem_data.return_code = ParameterizationReturnCode.ErrUnknown
        else:
            self._problem_data.return_code = ParameterizationReturnCode.Ok
        if return_data:
            return sdd_vec, sd_vec, v_vec, K
        else:
            return sdd_vec, sd_vec, v_vec

    def _one_step_forward(self, i, L_current, feasible_set_next):
        res = np.zeros(2)
        if np.isnan(L_current).any() or i < 0 or i > self._N:
            res[:] = np.nan
            return res
        nV = self.solver_wrapper.get_no_vars()
        g_upper = np.zeros(nV)
        deltas = self.solver_wrapper.get_deltas()[i - 1]
        g_upper[0] = -2 * deltas
        g_upper[1] = -1

        x_next_min = feasible_set_next[0]
        x_next_max = feasible_set_next[1]

        opt_1 = self.solver_wrapper.solve_stagewise_optim(
            i, None, g_upper, L_current[0], L_current[1], x_next_min, x_next_max
        )
        x_opt_1 = opt_1[1]
        u_opt_1 = opt_1[0]
        x_upper = x_opt_1 + 2 * deltas * u_opt_1

        opt_0 = self.solver_wrapper.solve_stagewise_optim(
            i, None, -g_upper, L_current[0], L_current[1], x_next_min, x_next_max
        )
        x_opt_0 = opt_0[1]
        u_opt_0 = opt_0[0]
        x_lower = x_opt_0 + 2 * deltas * u_opt_0

        res[:] = [x_lower, x_upper]
        return res

    def compute_reachable_sets(self, sdmin, sdmax):
        assert sdmin <= sdmax and 0 <= sdmin
        feasible_sets = self.compute_feasible_sets()
        L = np.zeros((self._N + 1, 2))
        L[0] = [sdmin ** 2, sdmax ** 2]
        logger.debug("Start computing the reachable sets")
        self.solver_wrapper.setup_solver()
        for i in range(0, self._N):
            L[i + 1] = self._one_step_forward(i, L[i], feasible_sets[i + 1])
            if L[i + 1, 0] < 0:
                L[i + 1, 0] = 0
            if np.isnan(L[i + 1]).any():
                logger.warn(
                    "L[{:d}]={:}. Path not parametrizable.".format(i + 1, L[i + 1])
                )
                return L
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "[Compute reachable sets] L_{:d}={:}".format(i + 1, L[i + 1])
                )

        self.solver_wrapper.close_solver()
        return L
