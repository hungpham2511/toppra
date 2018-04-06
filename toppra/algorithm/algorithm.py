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

    All algorithms should have three the following three attributes: `constraints`, `path`
    and `path_discretization`.

    All algorithms need to implement the method `compute_parameterization` as the least
    requirement to make the algorithm running.

    Parameters
    ----------
    constraint_list: list of `Constraint`
    path: `Interpolator`
        The geometric path, or the trajectory to parameterize.
    path_discretization: array, optional
        If not given, automatically generate a grid with 100 steps.
    """
    def __init__(self, constraint_list, path, path_discretization=None):
        if path_discretization is None:
            path_discretization = np.linspace(0, path.get_duration(), 100)
        self.constraints = constraint_list  # Attr
        self.path = path  # Attr
        self.path_discretization = np.array(path_discretization)  # Attr
        self._N = len(path_discretization) - 1  # Number of stages. Number of point is _N + 1
        assert path.get_path_interval()[0] == path_discretization[0]
        assert path.get_path_interval()[1] == path_discretization[-1]
        for i in range(self._N):
            assert path_discretization[i + 1] > path_discretization[i]

    def compute_parameterization(self, sd_start, sd_end):
        """ Compute a valid parameterization.

        If there is no valid parameterization, simply return None(s).

        Parameters
        ----------
        sd_start: float
            Starting path velocity. Must be positive.
        sd_end: float
            Goal path velocity. Must be positive.

        Returns
        -------
        sd_vec: (_N+1,) array None
            Path velocities.
        sdd_vec: (_N,) array or None
            Path accelerations.
        v_vec: (_N,) array or None
            Auxiliary variables.
        """
        raise NotImplementedError

    def compute_trajectory(self, sd_start, sd_end):
        """ Return the time-parameterized joint trajectory and auxilliary trajectory.

        Parameters
        ----------
        sd_start: float
            Starting path velocity.
        sd_end: float
            Goal path velocity.

        Returns
        -------
        Interpolator or None
            Time-parameterized joint position trajectory. If unable to parameterize, return None.
        Interpolator or None
            Time-parameterized auxiliary variable trajectory. If unable to
            parameterize or if there is no auxiliary variable, return None.
        """
        sdd_grid, sd_grid, v_grid = self.compute_parameterization(sd_start, sd_end)
        if sd_grid is None:
            return None, None

        # Gridpoint time instances
        t_grid = np.zeros(self._N + 1)
        for i in range(1, self._N + 1):
            sd_average = (sd_grid[i - 1] + sd_grid[i]) / 2
            delta_s = self.path_discretization[i] - self.path_discretization[i - 1]
            if sd_average > TINY:
                delta_t = delta_s / sd_average
            else:
                delta_t = 5  # If average speed is too slow.
            t_grid[i] = t_grid[i - 1] + delta_t

        q_grid = self.path.eval(self.path_discretization)
        traj_spline = SplineInterpolator(t_grid, q_grid)

        if v_grid.shape[1] == 0:
            v_spline = None
        else:
            v_grid_ = np.zeros((v_grid.shape[0] + 1, v_grid.shape[1]))
            v_grid_[:-1] = v_grid
            v_grid_[-1] = v_grid[-1]
            v_spline = SplineInterpolator(t_grid, v_grid_)

        return traj_spline, v_spline
