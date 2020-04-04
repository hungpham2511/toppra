from typing import List, Union, Optional
import numpy as np
from scipy.interpolate import BPoly


class SimplePath:
    """A standard class for representing continuous multi-dimentional function.

    Args:
     x: "Time instances" of the waypoints.
     y: Function values at waypoints.
     yd: First-derivative values.
     ydd: Second-derivative values.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, yd: np.ndarray = None, ydd=None):
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        self._polys = []
        for j in range(y.shape[1]):
            if yd is None:
                yd = np.zeros_like(y[:, j], dtype=float)
                for i in range(1, len(yd) - 1):
                    yd[i] = (y[i + 1, j] - y[i - 1, j]) / (x[i + 1] - x[i - 1])
            y_with_derivatives = np.vstack((y[:, j], yd)).T
            poly = BPoly.from_derivatives(x, y_with_derivatives)
            self._polys.append(poly)

    def __call__(self, xi, order=0):
        ret = []
        for poly in self._polys:
            if order == 1:
                val = poly.derivative()(xi)
            else:
                val = poly(xi)
            ret.append(val)
        return np.array(ret)
