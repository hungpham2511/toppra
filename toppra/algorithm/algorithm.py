"""
"""
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
    """Base class for all parameterization algorithms.

    All algorithms should have three attributes: `constraints`, `path`
    and `gridpoints` and also implement the method
    `compute_parameterization`.

    Parameters
    ----------
    constraint_list: list of `Constraint`
    path: `Interpolator`
        The geometric path, or the trajectory to parameterize.
    gridpoints: array, optional
        If not given, automatically generate a grid with 100 steps.
    """
    def __init__(self, constraint_list, path, gridpoints=None):
        if gridpoints is None:
            gridpoints = np.linspace(0, path.get_duration(), 100)
        self.constraints = constraint_list  # Attr
        self.path = path  # Attr
        self.gridpoints = np.array(gridpoints)  # Attr
        self._N = len(gridpoints) - 1  # Number of stages. Number of point is _N + 1
        assert path.get_path_interval()[0] == gridpoints[0]
        assert path.get_path_interval()[1] == gridpoints[-1]
        for i in range(self._N):
            assert gridpoints[i + 1] > gridpoints[i]

    def compute_parameterization(self, sd_start, sd_end):
        """ Compute a path parameterization.

        If there is no valid parameterization, simply return None(s).

        Parameters
        ----------
        sd_start: float
            Starting path velocity. Must be positive.
        sd_end: float
            Goal path velocity. Must be positive.

        Returns
        -------
        sdd_vec: (_N,) array or None
            Path accelerations.
        sd_vec: (_N+1,) array None
            Path velocities.
        v_vec: (_N,) array or None
            Auxiliary variables.
        """
        raise NotImplementedError

    def compute_trajectory(self, sd_start, sd_end, return_profile=False, bc_type='not-a-knot'):
        """ Compute the resulting joint trajectory and auxilliary trajectory.

        If parameterization fails, return a tuple of None(s).

        Parameters
        ----------
        sd_start: float
            Starting path velocity.
        sd_end: float
            Goal path velocity.

        Returns
        -------
        :class:`Interpolator`
            Time-parameterized joint position trajectory. If unable to parameterize, return None.
        :class:`Interpolator`
            Time-parameterized auxiliary variable trajectory. If unable to
            parameterize or if there is no auxiliary variable, return None.
        profiles: tuple
            Return if return_profile is True, results from compute_parameterization.
        """
        sdd_grid, sd_grid, v_grid, K = self.compute_parameterization(sd_start, sd_end, return_data=True)

        if sd_grid is None:
            if return_profile:
                return None, None, (sdd_grid, sd_grid, v_grid, K)
            else:
                return None, None

        # Gridpoint time instances
        t_grid = np.zeros(self._N + 1)
        for i in range(1, self._N + 1):
            sd_average = (sd_grid[i - 1] + sd_grid[i]) / 2
            delta_s = self.gridpoints[i] - self.gridpoints[i - 1]
            if sd_average > TINY:
                delta_t = delta_s / sd_average
            else:
                delta_t = 5  # If average speed is too slow.
            t_grid[i] = t_grid[i - 1] + delta_t

        q_grid = self.path.eval(self.gridpoints)
        traj_spline = SplineInterpolator(t_grid, q_grid, bc_type)

        if v_grid.shape[1] == 0:
            v_spline = None
        else:
            v_grid_ = np.zeros((v_grid.shape[0] + 1, v_grid.shape[1]))
            v_grid_[:-1] = v_grid
            v_grid_[-1] = v_grid[-1]
            v_spline = SplineInterpolator(t_grid, v_grid_)

        if return_profile:
            return traj_spline, v_spline, (sdd_grid, sd_grid, v_grid, K)
        else:
            return traj_spline, v_spline
