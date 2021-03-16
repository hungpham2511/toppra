"""
toppra.parametrizer
^^^^^^^^^^^^^^^^^^^^^

This module contains classes that produce the output trajectories,
given the input path and the time parametrization.

"""
import logging
import typing as T
import numpy as np
from toppra.interpolator import AbstractGeometricPath, SplineInterpolator
from toppra.exceptions import ToppraError
from toppra.constants import TINY

try:
    import matplotlib.pyplot as plt
except ImportError:
    pass

logger = logging.getLogger(__name__)

class ParametrizeConstAccel(AbstractGeometricPath):
    """Compute output traj under constant acceleration assumption.
    """

    def __init__(
        self,
        path: AbstractGeometricPath,
        gridpoints: np.ndarray,
        velocities: np.ndarray,
    ) -> None:
        self._path = path
        self._ss: np.ndarray = np.array(gridpoints)
        self._velocities: np.ndarray = np.array(velocities)
        self._xs: np.ndarray = self._velocities ** 2
        self._ts: T.Optional[np.ndarray] = None
        self._us: T.Optional[np.ndarray] = None

        # preconditions
        assert self._ss.shape[0] == self._velocities.shape[0]
        assert len(self._ss.shape) == 1
        assert np.all(self._velocities >= 0)

        self._process_parametrization()

    @property
    def dof(self) -> int:
        """Return the degrees-of-freedom of the path."""
        return self._path.dof

    def _process_parametrization(self):
        ts = [0]
        us = []
        for i in range(self._ss.shape[0] - 1):
            us.append(
                0.5 * (self._xs[i + 1] - self._xs[i]) / (self._ss[i + 1] - self._ss[i])
            )
            ts.append(
                ts[-1]
                + 2
                * (self._ss[i + 1] - self._ss[i])
                / (self._velocities[i] + self._velocities[i + 1])
            )
        self._ts = np.array(ts)
        self._us = np.array(us)

    @property
    def path_interval(self) -> np.ndarray:
        if self._ts is None:
            logger.warning("Unable to find _ts. Processing fails not does not run.")
            raise ValueError("Internal error occur.")
        return np.array([self._ts[0], self._ts[-1]])

    @property
    def duration(self) -> float:
        """Return the path duration."""
        return self.path_interval[1] - self.path_interval[0]

    def __call__(self, ts, order=0):
        scalar = False
        if isinstance(ts, (int, float)):
            ts = np.array([ts], dtype=float)
            scalar = True
        ss, vs, us = self._eval_params(ts)
        if order == 0:
            out = self._path(ss)
        elif order == 1:
            out = np.multiply(self._path(ss, 1), vs[:, np.newaxis])
        elif order == 2:
            out = (np.multiply(self._path(ss, 2), vs[:, np.newaxis] ** 2) +
                   np.multiply(self._path(ss, 1), us[:, np.newaxis]))
        else:
            raise ToppraError("Order %d is not supported." % order)
        if scalar:
            return out[0]
        return out

    def _eval_params(self, ts: np.ndarray) -> T.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the array of path positions, velocities and accels.

        Parameters
        -----------
        ts:
            Array of time instances.

        Returns
        ---------
        :
            Positions, velocity and accelerations.
        """
        assert self._us is not None
        assert self._ts is not None
        indices = np.searchsorted(self._ts, ts, side="right") - 1
        ss = []
        vs = []
        us = []
        for idx, t in zip(indices, ts):
            if idx == len(self._us):
                idx -= 1
            dt = t - self._ts[idx]
            us.append(self._us[idx])
            vs.append(self._velocities[idx] + dt * self._us[idx])
            ss.append(
                self._ss[idx]
                + dt * self._velocities[idx]
                + 0.5 * dt ** 2 * self._us[idx]
            )
        return np.array(ss), np.array(vs), np.array(us)

    def plot_parametrization(self, show: bool = False, n_sample: int = 500) -> None:
        """Plot the output parametrization and show it."""

        # small decrement to make sure all indices are valid
        ts = np.linspace(self.path_interval[0], self.path_interval[1], n_sample)
        ss, vs, _ = self._eval_params(ts)
        qs = self.__call__(ts, 0)

        plt.subplot(2, 2, 1)
        plt.plot(ts, ss, label="s(t)")
        plt.plot(self._ts, self._ss, "o", label="input")
        plt.title("path(time)")
        plt.legend()
        plt.subplot(2, 2, 2)
        plt.plot(ss, vs, label="v(s)")
        plt.plot(self._ss, self._velocities, "o", label="input")
        plt.title("velocity(path)")
        plt.legend()
        plt.subplot(2, 2, 3)
        plt.plot(ts, qs)
        plt.title("retimed path")
        plt.subplot(2, 2, 4)
        ss_dense = np.linspace(self._ss[0], self._ss[-1], n_sample)
        plt.plot(ss_dense, self._path(ss_dense))
        plt.title("original path")
        plt.tight_layout()
        if show:
            plt.show()


class ParametrizeSpline(SplineInterpolator):
    """Return output trajetory via sphine interpolation.

    This class computes the time and position at each gridpoint, then
    fit and return a CubicSpline (continuous first and second
    derivatives). Note that the boundary conditions are: first
    derivatives at the start and end of the path equal q(s)' * s'.

    """

    def __init__(self, path, gridpoints, velocities):
        # Gridpoint time instances
        t_grid = np.zeros_like(gridpoints)
        skip_ent = []
        for i in range(1, len(t_grid)):
            sd_average = (velocities[i - 1] + velocities[i]) / 2
            delta_s = gridpoints[i] - gridpoints[i - 1]
            if sd_average > TINY:
                delta_t = delta_s / sd_average
            else:
                delta_t = 5  # If average speed is too slow.
            t_grid[i] = t_grid[i - 1] + delta_t
            if delta_t < TINY:  # if a time increment is too small, skip.
                skip_ent.append(i)
        t_grid = np.delete(t_grid, skip_ent)
        gridpoints = np.delete(gridpoints, skip_ent)
        q_grid = path(gridpoints)

        super(ParametrizeSpline, self).__init__(
            t_grid,
            q_grid,
            (
                (1, path(path.path_interval[0], 1) * velocities[0]),
                (1, path(path.path_interval[1], 1) * velocities[-1]),
            ),
        )
