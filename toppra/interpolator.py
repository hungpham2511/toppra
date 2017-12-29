"""
This module contains several interfaces for interpolated path.
Most are simple wrappers over scipy.interpolators.
"""
import numpy as np
from scipy.interpolate import UnivariateSpline, CubicSpline


def normalize(ss):
    """ Normalize to one
    """
    return np.array(ss) / ss[-1]


class Interpolator(object):
    """ Abstract class for interpolators.
    """
    def __init__(self, ss_waypoints, waypoints):
        self.ss_waypoints = np.array(ss_waypoints)
        self.waypoints = np.array(waypoints)
        if np.isscalar(waypoints[0]):
            self.dof = 1
        else:
            self.dof = waypoints[0].shape[0]
        assert self.ss_waypoints.shape[0] == self.waypoints.shape[0]

    def get_waypoints(self):
        return self.ss_waypoints, self.waypoints

    def get_dof(self):
        return self.dof

    def get_path_interval(self):
        return np.array([self.ss_waypoints[0], self.ss_waypoints[1]])

    def eval(self, ss_sam):
        """ Evaluate positions.

        Parameters
        ----------
        ss_sam : array, or float
            Shape (m, ). Positions to sample at.

        Returns
        -------
        out : array
            Shape (m, dof). Evaluated values at position.
            Shape (dof,) if `ss_sam` is a float.
        """
        raise NotImplementedError

    def evald(self, ss_sam):
        """ Evaluate 1st derivative.

        Parameters
        ----------
        ss_sam : array
            Shape (m, ). Positions to sample at.

        Returns
        -------
        out : array
            Shape (m, dof). Evaluated values at position.
        """
        raise NotImplementedError

    def evaldd(self, ss_sam):
        """ Evaluate 2nd derivative.

        Parameters
        ----------
        ss_sam : array
            Shape (m, ). Positions to sample at.

        Returns
        -------
        out : array
            Shape (m, dof). Evaluated values at position.
        """
        raise NotImplementedError


class SplineInterpolator(Interpolator):
    """Interpolate the given waypoints by spline.

    This is a simple wrapper over scipy.CubicSpline class.

    Parameters
    ----------
    ss_waypoints: array
        Shaped (N+1,). Path positions of the waypoints.
    waypoints: array
        Shaped (N+1, dof). Waypoints.

    Attributes
    ----------
    dof : int
        Output dimension of the function
    cspl : :class:`scipy.interpolate.CubicSpline`
        The path.
    cspld : :class:`scipy.interpolate.CubicSpline`
        The path 1st derivative.
    cspldd : :class:`scipy.interpolate.CubicSpline`
        The path 2nd derivative.

    """
    def __init__(self, ss_waypoints, waypoints):
        super(SplineInterpolator, self).__init__(ss_waypoints, waypoints)

        self.cspl = CubicSpline(ss_waypoints, waypoints)
        self.cspld = self.cspl.derivative()
        self.cspldd = self.cspld.derivative()

    def eval(self, ss_sam):
        """ Evaluate positions.

        Parameters
        ----------
        ss_sam : array, or float
            Shape (m, ). Positions to sample at.

        Returns
        -------
        out : array
            Shape (m, dof). Evaluated values at position.
            Shape (dof,) if `ss_sam` is a float.
        """
        return self.cspl(ss_sam)

    def evald(self, ss_sam):
        """ Evaluate 1st derivative.

        Parameters
        ----------
        ss_sam : array
            Shape (m, ). Positions to sample at.

        Returns
        -------
        out : array
            Shape (m, dof). Evaluated values at position.
        """
        return self.cspld(ss_sam)

    def evaldd(self, ss_sam):
        """ Evaluate 2nd derivative.

        Parameters
        ----------
        ss_sam : array
            Shape (m, ). Positions to sample at.

        Returns
        -------
        out : array
            Shape (m, dof). Evaluated values at position.
        """
        return self.cspldd(ss_sam)


class UnivariateSplineInterpolator(Interpolator):
    """ Smooth given wayspoints by a cubic spline.

    This is a simple wrapper over scipy.UnivariateSplineInterpolator
    class.

    Parameters
    ----------
    ss: ndarray, shaped (N+1,)
        Path positions of the waypoints.
    qs: ndarray, shaped (N+1, dof)
        The waypoints.
    """
    def __init__(self, ss, qs):
        super(UnivariateSplineInterpolator, self).__init__(ss, qs)
        self.uspl = []
        for i in range(self.dof):
            self.uspl.append(UnivariateSpline(self.ss_waypoints, self.waypoints[:, i]))
        self.uspld = [spl.derivative() for spl in self.uspl]
        self.uspldd = [spl.derivative() for spl in self.uspld]

    def eval(self, ss):
        """ Evaluate positions.

        Parameters
        ----------
        ss_sam : array, or float
            Shape (m, ). Positions to sample at.

        Returns
        -------
        out : array
            Shape (m, dof). Evaluated values at position.
            Shape (dof,) if `ss_sam` is a float.
        """
        data = []
        for spl in self.uspl:
            data.append(spl(ss))
        return np.array(data).T

    def evald(self, ss):
        """ Evaluate 1st derivative.

        Parameters
        ----------
        ss_sam : array
            Shape (m, ). Positions to sample at.

        Returns
        -------
        out : array
            Shape (m, dof). Evaluated values at position.
        """
        data = []
        for spl in self.uspld:
            data.append(spl(ss))
        return np.array(data).T

    def evaldd(self, ss):
        """ Evaluate 2nd derivative.

        Parameters
        ----------
        ss_sam : array
            Shape (m, ). Positions to sample at.

        Returns
        -------
        out : array
            Shape (m, dof). Evaluated values at position.
        """
        data = []
        for spl in self.uspldd:
            data.append(spl(ss))
        return np.array(data).T


class PolyPath(object):
    """ A Polynomial Path class.
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
                for i in range(self.dof)]

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

