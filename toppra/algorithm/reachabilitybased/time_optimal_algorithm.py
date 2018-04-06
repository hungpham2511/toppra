from .reachability_algorithm import ReachabilityAlgorithm
import logging
import numpy as np

logger = logging.getLogger(__name__)


class TOPPRA(ReachabilityAlgorithm):
    """ The Time-Optimal Path Parameterization based on Reachability Analysis algorithm.

    """

    def _forward_step(self, i, x, K_next):
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

        Returns
        -------
        optim_var: array
            Optimal variable, which has this format (u, x, v).
        """
        if None in K_next or i < 0 or i > self._N:
            return [None, None]

        nV = self.solver_wrapper.get_no_vars()
        g_upper = np.zeros(nV)
        g_upper[1] = - 1
        g_upper[0] = - 2 * self.solver_wrapper.get_deltas()[i]

        optim_var = self.solver_wrapper.solve_stagewise_optim(
            i, None, g_upper, x, x, K_next[0], K_next[1])
        return optim_var

