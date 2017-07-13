import numpy as np
from scipy.interpolate import UnivariateSpline, CubicSpline


class PolynomialInterpolator(object):
    """
    """

    def __init__(self, coeff):
        self.coeff = np.array(coeff)
        if np.isscalar(self.coeff[0]):
            self.dof = 1
            self.poly = [np.polynomial.Polynomial(self.coeff)]
            self.coeff = self.coeff.reshape(1, -1)
        else:
            self.dof = self.coeff.shape[0]
            self.poly = [
                np.polynomial.Polynomial(self.coeff[i])
                for i in range(self.dof)
            ]

        self.polyd = [poly.deriv() for poly in self.poly]
        self.polydd = [poly.deriv() for poly in self.polyd]

    def eval(self, ss_sam):
        res = [poly(np.array(ss_sam)) for poly in self.poly]
        if self.dof == 1:
            return np.array(res).flatten()
        else:
            return np.array(res).T

    def evald(self, ss_sam):
        res = [poly(np.array(ss_sam)) for poly in self.polyd]
        if self.dof == 1:
            return np.array(res).flatten()
        else:
            return np.array(res).T

    def evaldd(self, ss_sam):
        res = [poly(np.array(ss_sam)) for poly in self.polydd]
        if self.dof == 1:
            return np.array(res).flatten()
        else:
            return np.array(res).T


def normalize(ss):
    """ Normalize to one
    """
    return np.array(ss) / ss[-1]


class SplineInterpolator(object):
    """Wrapper over scipy.CubicSpline

    Expose scipy.CubicSpline functionalities to the Interpolator interface.

    Args:
       ss: An ndarray. Shaped (N+1). Interpolated path position.
       qs: An ndarray. Shaped (N+1)xdof. Knotpoints jointvalues.
    """

    def __init__(self, ss, qs):
        """
        """
        # This class receives only normalized path position
        assert np.allclose(ss[-1], 1)
        self.cspl = CubicSpline(ss, qs)
        self.cspld = self.cspl.derivative()
        self.cspldd = self.cspld.derivative()
        if np.isscalar(qs[0]):
            self.dof = 1
        else:
            self.dof = qs[0].shape[0]

    def eval(self, ss_sam):
        return self.cspl(ss_sam)

    def evald(self, ss_sam):
        return self.cspld(ss_sam)

    def evaldd(self, ss_sam):
        return self.cspldd(ss_sam)


class UnivariateSplineInterpolator(object):
    """A smoother, not interpolator.

    This class does not interpolate via points, it smooths the input
    way points.

    """
    def __init__(self, ss, qs):
        """ All arguments are simiar to SplineInterpolator.
        """
        qs = np.array(qs)
        self.dof = qs.shape[1]
        self.uspl = []
        for i in range(self.dof):
            self.uspl.append(UnivariateSpline(ss, qs[:, i]))
        self.uspld = [spl.derivative() for spl in self.uspl]
        self.uspldd = [spl.derivative() for spl in self.uspld]

    def eval(self, ss):
        data = []
        for spl in self.uspl:
            data.append(spl(ss))
        return np.array(data).T

    def evald(self, ss):
        data = []
        for spl in self.uspld:
            data.append(spl(ss))
        return np.array(data).T

    def evaldd(self, ss):
        data = []
        for spl in self.uspldd:
            data.append(spl(ss))
        return np.array(data).T

