from typing import List, Union, Optional
import numpy as np
from scipy.interpolate import BPoly
from .interpolator import AbstractGeometricPath


class SimplePath(AbstractGeometricPath):
    """A class for representing continuous multi-dimentional function.

    This geometric path is specified by positions, velocities
    (optional) and acceleration (optional). Internally a scipy.PPoly
    instance is used to store the path. The polynomial degree depends
    on the input.

    If velocity is not given, they will be computed automatically.

    Parameters
    ------------
     x:
       "Time instances" of the waypoints.
     y:
       Function values at waypoints.
     yd:
       First-derivatives. If not given (None) will be computed
       automatically.

    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        yd: np.ndarray = None,
    ):
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        if yd is not None and len(yd.shape) == 1:
            yd = yd.reshape(-1, 1)
        self._x = x
        self._y = y.astype(float)
        self._yd = yd if yd is None else yd.astype(float)
        self._polys = self._construct_polynomials()

    def __call__(self, xi, order=0):
        """Evaluate the path at given position."""
        ret = [poly.derivative(order)(xi) for poly in self._polys]
        return np.array(ret).T

    @property
    def dof(self):
        return self._y.shape[1]

    @property
    def path_interval(self):
        return np.array([self._x[0], self._x[-1]], dtype=float)

    @property
    def waypoints(self):
        return self._y

    def _autofill_yd(self):
        if self._yd is None:
            _yd = np.zeros_like(self._y[:], dtype=float)
            for i in range(1, len(_yd) - 1):
                _yd[i] = (self._y[i + 1] - self._y[i - 1]) / (
                    self._x[i + 1] - self._x[i - 1]
                )
        else:
            _yd = np.array(self._yd[:])
        return _yd

    def _construct_polynomials(self):
        polys = []
        _yd = self._autofill_yd()

        for j in range(self._y.shape[1]):

            y_with_derivatives = np.vstack((self._y[:, j], _yd[:, j])).T
            poly = BPoly.from_derivatives(self._x, y_with_derivatives)
            polys.append(poly)

        return polys
