"""
This module contains several interfaces for interpolated path.
Most are simple wrappers over scipy.interpolators.
"""
import numpy as np
from scipy.interpolate import UnivariateSpline, CubicSpline


def normalize(ss):
    """ Normalize the path discretization.

    Parameters
    ----------
    ss: ndarray
        Path position array.

    Returns
    -------
    out: ndarray
        Normalized path position array.
    """
    return np.array(ss) / ss[-1]


def _find_left_index(ss_waypoints, s):
    """Return the index of the largest entry in `ss_waypoints` that is
    larger or equal `s`.

    Parameters
    ----------
    ss_waypoints: ndarray
        Array of path positions.
    s: float
        A single path position.

    Returns
    -------
    out: int
        The desired index.
    """
    for i in range(1, len(ss_waypoints)):
        if ss_waypoints[i - 1] <= s and s < ss_waypoints[i]:
            return i - 1
    return len(ss_waypoints) - 2


class Interpolator(object):
    """ Abstract class for interpolators.
    """
    def __init__(self):
        self.dof = None
        self.duration = None

    def get_dof(self):
        """ Return the degree-of-freedom of the path.

        Returns
        -------
        out: int
            Degree-of-freedom of the path.
        """
        raise NotImplementedError

    def get_duration(self):
        """

        Returns
        -------

        """
        raise NotImplementedError

    def get_path_interval(self):
        """ Return the starting and ending path positions.

        Returns
        -------
        out: ndarray
            Shaped (2,).

        """
        return np.array([self.s_start, self.s_end])

    def eval(self, ss_sam):
        """ Evaluate joint positions at specified path positions.

        Parameters
        ----------
        ss_sam : array, or float
            Shape (m, ). Path positions to sample at.

        Returns
        -------
        out : array
            Shape (m, dof). Evaluated values at position.
            Shape (dof,) if `ss_sam` is a float.
        """
        raise NotImplementedError

    def evald(self, ss_sam):
        position__ = """ Evaluate the first derivative of the geometric path.

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
        """ Evaluate the 2nd derivative of the geometric path.

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

    def serialize_rave(self):
        raise NotImplementedError

    def serialize_ros(self):
        raise NotImplementedError


class RaveTrajectoryWrapper(Interpolator):
    """A wrapper over OpenRAVE's :class:`GenericTrajectory`.

    Parameters
    ----------
    traj: :class:`openravepy.GenericTrajectory`
        An OpenRAVE joint trajectory using quadratic interpolation.
    robot: :class:`openravepy.GenericRobot`
        An OpenRAVE robot, suitable to `traj`.

    Notes
    -----
    Only trajectories using quadratic interpolation is
    supported. Supports for other kinds will be added in the future.
    """
    def __init__(self, traj, robot):
        self.traj = traj  #: init
        self.spec = traj.GetConfigurationSpecification()
        self.dof = robot.GetActiveDOF()

        assert self.spec.GetGroupFromName('joint').interpolation == 'quadratic', "This class only handle trajectory with quadratic interpolation"

        self.n_waypoints = traj.GetNumWaypoints()
        dt_waypoints = [self.spec.ExtractDeltaTime(traj.GetWaypoint(i))
                        for i in range(self.n_waypoints)]
        self.ss_waypoints = np.array(dt_waypoints)
        self.s_start = self.ss_waypoints[0]
        self.s_end = self.ss_waypoints[1]
        for i in range(1, self.n_waypoints):
            self.ss_waypoints[i] = dt_waypoints[i] + self.ss_waypoints[i - 1]

        self.waypoints = [self.spec.ExtractJointValues(traj.GetWaypoint(i), robot, robot.GetActiveDOFIndices())
                          for i in range(self.n_waypoints)]
        self.waypoints_deriv = [self.spec.ExtractJointValues(traj.GetWaypoint(i), robot, robot.GetActiveDOFIndices(), 1)
                                for i in range(self.n_waypoints)]
        self.waypoints_dderiv = []
        for i in range(self.n_waypoints - 1):
            qdd = ((self.waypoints_deriv[i + 1] - self.waypoints_deriv[i])
                   / dt_waypoints[i + 1])
            self.waypoints_dderiv.append(qdd)

    def eval(self, ss_sam):
        if np.isscalar(ss_sam):
            index = _find_left_index(self.ss_waypoints, ss_sam)
            qdd_left = self.waypoints_dderiv[index]
            qd_left = self.waypoints_deriv[index]
            q_left = self.waypoints[index]
            ds = (ss_sam - self.ss_waypoints[index])
            q = q_left + qd_left * ds + qdd_left * ds ** 2 / 2
            return q
        else:
            qs = []
            for s in ss_sam:
                index = _find_left_index(self.ss_waypoints, s)
                qdd_left = self.waypoints_dderiv[index]
                qd_left = self.waypoints_deriv[index]
                q_left = self.waypoints[index]
                ds = (s - self.ss_waypoints[index])
                q = q_left + qd_left * ds + qdd_left * ds ** 2 / 2
                qs.append(q)
            return np.array(qs)

    def evald(self, ss_sam):
        if np.isscalar(ss_sam):
            index = _find_left_index(self.ss_waypoints, ss_sam)
            qdd_left = self.waypoints_dderiv[index]
            qd_left = self.waypoints_deriv[index]
            deltat = (ss_sam - self.ss_waypoints[index])
            qd = qd_left + qdd_left * deltat
            return qd
        else:
            qds = []
            for s in ss_sam:
                index = _find_left_index(self.ss_waypoints, s)
                qdd_left = self.waypoints_dderiv[index]
                qd_left = self.waypoints_deriv[index]
                deltat = (s - self.ss_waypoints[index])
                qd = qd_left + qdd_left * deltat
                qds.append(qd)
            return np.array(qds)

    def evaldd(self, ss_sam):
        if np.isscalar(ss_sam):
            index = _find_left_index(self.ss_waypoints, ss_sam)
            qdd_left = self.waypoints_dderiv[index]
            return qdd_left
        else:
            qdds = []
            for s in ss_sam:
                index = _find_left_index(self.ss_waypoints, s)
                qdd_left = self.waypoints_dderiv[index]
                qdds.append(qdd_left)
            return np.array(qdds)


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
        super(SplineInterpolator, self).__init__()
        assert ss_waypoints[0] == 0, "First index must equals zero."
        self.ss_waypoints = np.array(ss_waypoints)
        self.waypoints = np.array(waypoints)
        if np.isscalar(waypoints[0]):
            self.dof = 1
        else:
            self.dof = waypoints[0].shape[0]
        self.duration = ss_waypoints[-1]
        assert self.ss_waypoints.shape[0] == self.waypoints.shape[0]
        self.s_start = self.ss_waypoints[0]
        self.s_end = self.ss_waypoints[-1]

        self.cspl = CubicSpline(ss_waypoints, waypoints)
        self.cspld = self.cspl.derivative()
        self.cspldd = self.cspld.derivative()

    def get_duration(self):
        return self.duration

    def eval(self, ss_sam):
        return self.cspl(ss_sam)

    def evald(self, ss_sam):
        return self.cspld(ss_sam)

    def evaldd(self, ss_sam):
        return self.cspldd(ss_sam)


class UnivariateSplineInterpolator(Interpolator):
    """ Smooth given wayspoints by a cubic spline.

    This is a simple wrapper over `scipy.UnivariateSplineInterpolator`
    class.

    Parameters
    ----------
    ss_waypoints: ndarray
        Path positions of the waypoints.
    waypoints: ndarray
        The waypoints.
    """
    def __init__(self, ss_waypoints, waypoints):
        super(UnivariateSplineInterpolator, self).__init__()
        assert ss_waypoints[0] == 0, "First index must equals zero."
        self.ss_waypoints = np.array(ss_waypoints)
        self.waypoints = np.array(waypoints)
        if np.isscalar(waypoints[0]):
            self.dof = 1
        else:
            self.dof = waypoints[0].shape[0]
        self.duration = ss_waypoints[-1]
        assert self.ss_waypoints.shape[0] == self.waypoints.shape[0]
        self.s_start = self.ss_waypoints[0]
        self.s_end = self.ss_waypoints[-1]

        self.uspl = []
        for i in range(self.dof):
            self.uspl.append(UnivariateSpline(self.ss_waypoints, self.waypoints[:, i]))
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


class PolynomialPath(Interpolator):
    """ A class representing polynominal paths.

    If coeff is a 1d array, the polynomial's equation is given by

    .. math::

    coeff[0] + coeff[1] s + coeff[2] s^2 + ...

    If coeff is a 2d array, the i-th joint position is the polynomial

    .. math::

    coeff[i, 0] + coeff[i, 1] s + coeff[i, 2] s^2 + ...

    Parameters
    ----------
    coeff : ndarray
        Coefficients of the polynomials.
    s_start: float, optional
        Starting path position.
    s_end: float, optional
        Goal path position.
    """
    def __init__(self, coeff, s_start=0, s_end=1):
        self.coeff = np.array(coeff)
        self.s_start = s_start
        self.s_end = s_end
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
