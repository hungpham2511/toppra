from .reachability_algorithm import ReachabilityAlgorithm
import logging
import numpy as np

logger = logging.getLogger(__name__)


class TOPPRA(ReachabilityAlgorithm):
    """Time-Optimal Path Parameterization based on Reachability
    Analysis (TOPPRA).


    Parameters
    ----------
    constraint_list: :class:`~toppra.constraint.Constraint` list
        List of constraints to which the robotic system is subjected to.
    path: :class:`~toppra.Interpolator`
        Input geometric path.
    gridpoints: array, optional
        Gridpoints for discretization of the geometric path. The start
        and end points must agree with the geometric path's domain.
    solver_wrapper: str, optional
        Name of the solver wrapper to use.
        See :class:`toppra.solverwrapper.hotqpOASESSolverWrapper`,
        :class:`toppra.solverwrapper.seidelWrapper`

    Notes
    -----
    In addition to the input constraints, there might be additional
    constraints on the solutions. These constraints usually come from
    the solver wrapper. Therefore, don't be suprise if different
    parametrizations are returned for different solver
    wrappers. However, the different should be very small, especially
    for well-conditioned problems.

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
        eps: float, optional
            A numerical constant to avoid propagating numerical errors.

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
        g_upper[1] = - 1
        g_upper[0] = - 2 * self.solver_wrapper.get_deltas()[i]

        # Account for propagating numerical errors
        K_next_max = K_next[1]
        K_next_min = K_next[0]

        optim_var = self.solver_wrapper.solve_stagewise_optim(
            i, None, g_upper, x, x, K_next_min, K_next_max)
        return optim_var

