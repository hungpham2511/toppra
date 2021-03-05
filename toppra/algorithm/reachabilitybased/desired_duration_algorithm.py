from .reachability_algorithm import ReachabilityAlgorithm
from ...constants import LARGE, SMALL
from .. import algorithm as algo
import logging
import numpy as np

logger = logging.getLogger(__name__)


def _compute_duration(xs, deltas):
    """ Compute the duration of the given parametrization.
    """
    sds = np.sqrt(xs)
    t = 0
    for i in range(len(deltas)):
        t += 2 * deltas[i] / (sds[i + 1] + sds[i] + 1e-9)
    return t


class TOPPRAsd(ReachabilityAlgorithm):
    """TOPPRA with specified duration.

    The key technical idea is to compute the **fastest** and the
    **slowest** time parameterizations. Then use bisection search to
    find a convex combination of the parametrizations that has the
    desired duration.

    TODO: The current implementation is inefficient.

    """

    def set_desired_duration(self, desired_duration: float):
        """Set desired duration for the time-parametrization.

        Parameters
        ----------
        desired_duration: 
          The desired duration.
        """
        self.desired_duration = desired_duration

    def compute_parameterization(self, sd_start: float, sd_end: float, return_data: bool=False, atol: float=1e-5):
        """Compute a path parameterization.

        If there is no valid parameterizations, simply return None(s).
        If the desired duration is not achievable, returns the fastest
        or the slowest parameterizations.

        Parameters
        ----------
        sd_start:
            Starting path velocity. Must be positive.
        sd_end:
            Goal path velocity. Must be positive.
        return_data:
            If is True, also return matrix K which contains the controllable sets. Default to False.
        atol:
            Absolute tolerance of duration. Default to 1e-5.

        Returns
        -------
        sdd_vec: array or None
            Shape (N,). Path accelerations.
        sd_vec: array None
            Shape (N+1,). Path velocities.
        v_vec: array or None
            Shape (N,). Auxiliary variables.
        K: array
            Shape (N+1, 2). Return if `return_data` is True. The controllable sets.

        """
        assert sd_end >= 0 and sd_start >= 0, "Path velocities must be positive"
        K = self.compute_controllable_sets(sd_end, sd_end)
        if np.isnan(K).any():
            logger.warn("The set of controllable velocities at the beginning is empty!")
            self._problem_data.return_code = algo.ParameterizationReturnCode.FailUncontrollable
            if return_data:
                return None, None, None, K
            else:
                return None, None, None
        self.problem_data.K = K

        x_start = sd_start ** 2
        if x_start + SMALL < K[0, 0] or K[0, 1] + SMALL < x_start:
            logger.warn("The initial velocity is not controllable. {:f} not in ({:f}, {:f})".format(
                x_start, K[0, 0], K[0, 1]
            ))
            self._problem_data.return_code = algo.ParameterizationReturnCode.FailUncontrollable
            if return_data:
                return None, None, None, K
            else:
                return None, None, None
        N = self.solver_wrapper.get_no_stages()
        deltas = self.solver_wrapper.get_deltas()
        # compute the fastest parametrization
        xs = np.zeros(N + 1)
        xs[0] = x_start
        us = np.zeros(N)
        v_vec = np.zeros((N, self.solver_wrapper.get_no_vars() - 2))

        self.solver_wrapper.setup_solver()
        for i in range(self._N):
            optim_res = self._forward_step(i, xs[i], K[i + 1])
            if np.isnan(optim_res[0]):
                logger.fatal("A numerical error occurs: The instance is controllable "
                             "but forward pass fails.")
                us[i] = np.nan
                xs[i + 1] = np.nan
                v_vec[i] = np.nan
            else:
                us[i] = optim_res[0]
                # The below function min( , max( ,)) ensure that the state x_{i+1} is controllable.
                # While this is ensured theoretically by the existence of the controllable sets,
                # numerical errors might violate this condition.
                xs[i + 1] = min(K[i + 1, 1], max(K[i + 1, 0], xs[i] + 2 * deltas[i] * us[i] - SMALL))
                v_vec[i] = optim_res[2:]
            logger.debug("[Forward pass] u_{:d} = {:f}, x_{:d} = {:f}".format(i, us[i], i + 1, xs[i + 1]))
        self.solver_wrapper.close_solver()
        # compute the slowest parametrization
        xs_slow = np.zeros(N + 1)
        xs_slow[0] = x_start
        us_slow = np.zeros(N)
        v_vec_slow = np.zeros((N, self.solver_wrapper.get_no_vars() - 2))

        self.solver_wrapper.setup_solver()
        for i in range(self._N):
            optim_res = self._forward_step(i, xs_slow[i], K[i + 1], slow=True)
            if np.isnan(optim_res[0]):
                logger.fatal("A numerical error occurs: The instance is controllable "
                             "but forward pass fails.")
                us_slow[i] = np.nan
                xs_slow[i + 1] = np.nan
                v_vec_slow[i] = np.nan
            else:
                us_slow[i] = optim_res[0]
                # The below function min( , max( ,)) ensure that the state x_{i+1} is controllable.
                # While this is ensured theoretically by the existence of the controllable sets,
                # numerical errors might violate this condition.
                xs_slow[i + 1] = min(K[i + 1, 1], max(K[i + 1, 0], xs_slow[i] + 2 * deltas[i] * us_slow[i] - SMALL))
                v_vec_slow[i] = optim_res[2:]
            logger.debug("[Forward pass] u_{:d} = {:f}, x_{:d} = {:f}".format(i, us_slow[i], i + 1, xs_slow[i + 1]))
        self.solver_wrapper.close_solver()

        # desired parametrization xs_desired = alpha * xs + (1 - alpha) / xs_slow
        duration = _compute_duration(xs, deltas)
        duration_slow = _compute_duration(xs_slow, deltas)
        if duration > self.desired_duration:
            logger.warn("Desired duration {:f} seconds is not achievable."
                        " Returning the fastest parameterization with duration {:f} seconds"
                        "".format(self.desired_duration, duration))
            alpha = 1.0
        elif duration_slow < self.desired_duration:
            logger.warn("Desired duration {:f} seconds is not achievable."
                        " Returning the slowest parameterization with duration {:f} seconds"
                        "".format(self.desired_duration, duration))
            alpha = .0
        else:
            logger.info("Desired duration {:f} sec is achievable. Continue computing.".format(self.desired_duration))
            alpha_low = 1.0  # here, low means a lower duration, and faster speed
            alpha_high = 0.0
            diff = 10
            it = 0
            while diff > atol:
                it += 1
                alpha = 0.5 * (alpha_low + alpha_high)
                xs_alpha = alpha * xs + (1 - alpha) * xs_slow
                duration_alpha = _compute_duration(xs_alpha, deltas)
                if duration_alpha < self.desired_duration:
                    alpha_low = alpha
                    diff = self.desired_duration - duration_alpha
                else:  # duration_alpha >= self.desired_duration
                    alpha_high = alpha
                    diff = duration_alpha - self.desired_duration
                logger.debug("it: {:d}. search range: [{:}, {:}]".format(
                    it, alpha_low, alpha_high))
        xs_alpha = alpha * xs + (1 - alpha) * xs_slow
        us_alpha = alpha * us + (1 - alpha) * us_slow
        v_vec_alpha = alpha * v_vec + (1 - alpha) * v_vec_slow

        sd_vec = np.sqrt(xs_alpha)
        sdd_vec = np.copy(us_alpha)
        self.problem_data.sd_vec = sd_vec
        self.problem_data.sdd_vec = sdd_vec
        if np.isnan(sd_vec).any():
            self.problem_data.return_code = algo.ParameterizationReturnCode.ErrUnknown
        else:
            self.problem_data.return_code = algo.ParameterizationReturnCode.Ok
        if return_data:
            return sdd_vec, sd_vec, v_vec_alpha, K
        else:
            return sdd_vec, sd_vec, v_vec_alpha

    def _forward_step(self, i, x, K_next, slow=False):
        """ Compute the highest possible path velocity that is controllable.

        Parameters
        ----------
        i: int
            Current stage index
        x: float
            The squared velocity at the current stage.
        K_next: list
            The lower and upper bounds of the set of controllable squared velocities
            in the next stage.
        eps: float, optional
            A numerical constant to avoid propagating numerical errors.
        slow: bool, optional

        Returns
        -------
        optim_var: array
            Optimal variable, which has this format (u, x, v).
            If this step fails, `optim_var` contains only nans.
        """
        # Immediate return
        if None in K_next or i < 0 or i > self._N or np.isnan(x) or x is None:
            return np.array([np.nan, np.nan])

        nV = self.solver_wrapper.get_no_vars()
        g_upper = np.zeros(nV)
        if not slow:
            g_upper[1] = - 1
            g_upper[0] = - 2 * self.solver_wrapper.get_deltas()[i]
        else:
            g_upper[1] = 1
            g_upper[0] = 2 * self.solver_wrapper.get_deltas()[i]

        # Account for propagating numerical errors
        K_next_max = K_next[1]
        K_next_min = K_next[0]

        optim_var = self.solver_wrapper.solve_stagewise_optim(
            i, None, g_upper, x, x, K_next_min, K_next_max)
        return optim_var
