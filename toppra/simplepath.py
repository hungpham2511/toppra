from typing import List, Union
import numpy as np
from scipy.interpolate import BPoly


class SimplePath:

    def __init__(self, x: List[float], y: Union[List[float], np.ndarray], yd=None, ydd=None):
        if yd is None:
            yd = np.zeros_like(x, dtype=float)
            for i in range(1, len(y) - 1):
                yd[i] = (y[i + 1] - y[i - 1]) / (x[i + 1] - x[i - 1])
        y = np.vstack((y, yd)).T

        self._poly = BPoly.from_derivatives(x, y)

    def __call__(self, xi, order=0):
        if order == 1:
            return self._poly.derivative()(xi)
        return self._poly(xi)
