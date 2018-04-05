import numpy as np

try:
    import openravepy as orpy
    OPENRAVEPY_AVAILABLE = True
except ImportError:
    OPENRAVEPY_AVAILABLE = False

from ..constants import TINY
from ..interpolator import SplineInterpolator

import logging
logger = logging.getLogger(__name__)


class ParameterizationAlgorithm(object):
    """
    Base class for all parameterization algorithms.

    Parameters
    ----------
    constraint_list: list of `Constraint`
    path: `Interpolator`
    path_discretization: array, optional

    """

    def __init__(self, constraint_list, path, path_discretization=None):
        if path_discretization is None:
            path_discretization = np.linspace(0, path.get_duration(), 100)
        self.N = len(path_discretization) - 1  # Number of stages. Number of point is N + 1
        self.path = path
        self.constraints = constraint_list
        self.path_discretization = np.array(path_discretization)
        assert path.get_path_interval()[0] == path_discretization[0]
        assert path.get_path_interval()[1] == path_discretization[-1]
        for i in range(self.N):
            assert path_discretization[i + 1] > path_discretization[i]

    def compute_parameterization(self, sd_start, sd_end):
        raise NotImplementedError

    def compute_trajectory(self, sd_start, sd_end):
        """ Return a trajectory sampled at the grid points.

        Parameters
        ----------
        sd_start: float
            Starting path velocity.
        sd_end: float
            Goal path velocity.

        Returns
        -------
        t_grid: array
            Time instances.
        q_grid: array
            Joint positions.
        qd_vec: array
            Joint velocities.
        qdd_vec: array
            Joint accelerations.

        Notes
        -----
        Result contains the time instance, joint position, velocity and acceleration at
        each point of the path discretization grid. The following formula are used:

        t[i] = t[i-1] + 2 * (s[i] - s[i-1]) / (sd[i] + sd[i+1])
        q[i] = p[i]
        qd[i] = ps[i] * sd[i]
        qdd[i] = ps[i] * sdd[i] + pss[i] * sd[i] ^ 2
        """
        sd_grid, sdd_grid = self.compute_parameterization(sd_start, sd_end)
        if sd_grid is None:
            return None, None, None, None

        # Gridpoint time instances
        t_grid = np.zeros(self.N + 1)
        for i in range(1, self.N + 1):
            sd_average = (sd_grid[i] + sd_grid[i + 1]) / 2
            delta_s = self.path_discretization[i + 1] - self.path_discretization[i]
            if sd_average > TINY:
                delta_t = delta_s / sd_average
            else:
                delta_t = 5  # If average speed is too slow.
            t_grid[i] = t_grid[i - 1] + delta_t

        q_grid = self.path.eval(self.path_discretization)

        traj_spline = SplineInterpolator(t_grid, q_grid)
        return traj_spline
