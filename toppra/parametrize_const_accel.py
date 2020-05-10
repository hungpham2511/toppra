from typing import Tuple
import numpy as np
from toppra.interpolator import AbstractGeometricPath
import matplotlib.pyplot as plt


class ParametrizeConstAccel(AbstractGeometricPath):
    def __init__(self, path, gridpoints, velocities):
        self._path = path
        self._ss: np.ndarray = np.array(gridpoints)
        self._velocities: np.ndarray = np.array(velocities)
        self._xs = self._velocities ** 2
        self._ts = None
        self._us = None
        
        # preconditions
        assert(self._ss.shape[0] == self._velocities.shape[0])
        assert(len(self._ss.shape) == 1)
        assert(np.all(self._velocities >= 0))

        self._process_parametrization()

    def _process_parametrization(self):
        ts = [0]
        us = []
        for i in range(self._ss.shape[0] - 1):
            us.append(0.5 * (self._xs[i + 1] - self._xs[i]) / (self._ss[i + 1] - self._ss[i]))
            ts.append(ts[-1] + 2 * (self._ss[i + 1] - self._ss[i]) / (self._velocities[i] + self._velocities[i + 1]))
        self._ts = np.array(ts)
        self._us = np.array(us)

    @property
    def path_interval(self):
        return np.array([self._ts[0], self._ts[-1]])

    def _eval_params(self, ts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the array of path positions, velocities and accels."""
        indices = np.searchsorted(self._ts, ts, side='right') - 1
        ss = []
        vs = []
        us = []
        for idx, t in zip(indices, ts):
            dt = t - self._ts[idx]
            us.append(self._us[idx])
            vs.append(self._velocities[idx] + dt * self._us[idx])
            ss.append(self._ss[idx] + dt * self._velocities[idx] + 0.5 * dt ** 2 * self._us[idx])
        return np.array(ss), np.array(vs), np.array(us)

    def plot_parametrization(self, show=False, n_sample=500):
        # small decrement to make sure all indices are valid
        ts = np.linspace(self.path_interval[0], self.path_interval[1]-1e-6, n_sample)
        ss, vs, us = self._eval_params(ts)
        plt.subplot(1, 2, 1)
        plt.plot(ts, ss, label='s(t)')
        plt.plot(self._ts, self._ss, 'o', label='input')
        plt.title('path(time)')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(ss, vs, label='v(s)')
        plt.plot(self._ss, self._velocities, 'o', label='input')
        plt.title('velocity(path)')
        plt.legend()
        if show:
            plt.show()

            
