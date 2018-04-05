from .reachability_algorithm import ReachabilityAlgorithm
import logging
import numpy as np

logger = logging.getLogger(__name__)

class TOPPRA(ReachabilityAlgorithm):
    """ The Time-Optimal Path Parameterization based on Reachability Analysis algorithm.

    """
    def compute_parameterization(self, sd_start, sd_end):
        """

        Parameters
        ----------
        sd_start
        sd_end

        Returns
        -------
        sd_grid
        sdd_grid

        """
        assert sd_end >= 0 and sd_start >= 0, "Path velocities must be positive"
        K = self.compute_controllable_sets(sd_end, sd_end)

        if None in K[0]:
            logger.warn("The set of controllable velocities at the beginning is empty!")
            return None, None

        x_start = sd_start ** 2
        if x_start < K[0, 0] or K[0, 1] < x_start:
            logger.warn("The initial velocity is not controllable.")
            return None, None

        N = self.solver_wrapper.get_no_stages()
        xs = np.zeros(N + 1)
        xs[0] = x_start
        us = np.zeros(N)
        # for i in range(N):



