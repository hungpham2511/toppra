"""
This module contains several interfaces for interpolated path.
Most are simple wrappers over scipy.interpolators.
"""
import numpy as np
from scipy.interpolate import UnivariateSpline, CubicSpline, PPoly
import logging
logger = logging.getLogger(__name__)
try:
    import openravepy as orpy
except ImportError:
    logger.warn("Openravepy not found!")


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
        # Note: do not use this attribute directly, use get_duration
        # method instead.
        self.duration = None
        self.scaling = 1.0

    def get_dof(self):
        """ Return the degree-of-freedom of the path.

        Returns
        -------
        out: int
            Degree-of-freedom of the path.
        """
        return self.dof

    def set_scaling(self, scaling):
        """Scale the path.

        The if the original path duration is [0, s_end], then the
        scaled duration is [0, s_end * gamma]. Remark that a scaled
        path also have scaled derivatives. Using bar to denoted the
        scaled path, by simple calculus one has:

        duration_bar = duration * gamma
        q_bar(s_bar) = q(s_bar / gamma)
        dq_bar(s_bar) / ds_bar = dq(s_bar / gamma) / ds / gamma
        d2q_bar(s_bar) / ds_bar^2 = d2q(s_bar / gamma) / ds^2 / gamma^2
        """
        assert scaling > 0, "scaling must be a positive float!"
        self.scaling = float(scaling)

    def get_duration(self):
        """ Return the duration of the path.

        Returns
        -------
        out: float
            Path duration.

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
        """ Evaluate the first derivative of the geometric path.

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

    def compute_rave_trajectory(self):
        raise NotImplementedError

    def compute_ros_trajectory(self):
        raise NotImplementedError


class RaveTrajectoryWrapper(Interpolator):
    """An interpolator that wraps OpenRAVE's :class:`GenericTrajectory`.

    Only trajectories using quadratic interpolation or cubic interpolation are supported.
    The trajectory is represented as a piecewise polynomial. The polynomial could be
    quadratic or cubic depending the interpolation method used by the input trajectory object.

    Parameters
    ----------
    traj: :class:`openravepy.GenericTrajectory`
        An OpenRAVE joint trajectory.
    robot: :class:`openravepy.GenericRobot`
        An OpenRAVE robot.
    """
    def __init__(self, traj, robot):
        self.traj = traj  #: init
        self.spec = traj.GetConfigurationSpecification()
        self.dof = robot.GetActiveDOF()

        self._interpolation = self.spec.GetGroupFromName('joint').interpolation
        assert self._interpolation == 'quadratic' or self._interpolation == "cubic", "This class only handles trajectories with quadratic or cubic interpolation"
        self._duration = traj.GetDuration()
        all_waypoints = traj.GetWaypoints(0, traj.GetNumWaypoints()).reshape(traj.GetNumWaypoints(), -1)
        valid_wp_indices = [0]
        self.ss_waypoints = [0.0]
        for i in range(1, traj.GetNumWaypoints()):
            dt = self.spec.ExtractDeltaTime(all_waypoints[i])
            if dt > 1e-5:  # If delta is too small, skip it.
                valid_wp_indices.append(i)
                self.ss_waypoints.append(self.ss_waypoints[-1] + dt)

        self.n_waypoints = len(valid_wp_indices)
        self.ss_waypoints = np.array(self.ss_waypoints)
        self.s_start = self.ss_waypoints[0]
        self.s_end = self.ss_waypoints[-1]

        self.waypoints = np.array([self.spec.ExtractJointValues(all_waypoints[i], robot, robot.GetActiveDOFIndices()) for i in valid_wp_indices])
        self.waypoints_d = np.array([self.spec.ExtractJointValues(all_waypoints[i], robot, robot.GetActiveDOFIndices(), 1) for i in valid_wp_indices])

        # Degenerate case: there is only one waypoint.
        if self.n_waypoints == 1:
            pp_coeffs = np.zeros((1, 1, self.dof))
            for idof in range(self.dof):
                pp_coeffs[0, 0, idof] = self.waypoints[0, idof]
            # A constant function
            self.ppoly = PPoly(pp_coeffs, [0, 1])

        elif self._interpolation == "quadratic":
            self.waypoints_dd = []
            for i in range(self.n_waypoints - 1):
                qdd = ((self.waypoints_d[i + 1] - self.waypoints_d[i]) / (self.ss_waypoints[i + 1] - self.ss_waypoints[i]))
                self.waypoints_dd.append(qdd)
            self.waypoints_dd = np.array(self.waypoints_dd)

            # Fill the coefficient matrix for scipy.PPoly class
            pp_coeffs = np.zeros((3, self.n_waypoints - 1, self.dof))
            for idof in range(self.dof):
                for iseg in range(self.n_waypoints - 1):
                    pp_coeffs[:, iseg, idof] = [self.waypoints_dd[iseg, idof] / 2,
                                                self.waypoints_d[iseg, idof],
                                                self.waypoints[iseg, idof]]
            self.ppoly = PPoly(pp_coeffs, self.ss_waypoints)

        elif self._interpolation == "cubic":
            self.waypoints_dd = np.array([self.spec.ExtractJointValues(all_waypoints[i], robot, robot.GetActiveDOFIndices(), 2) for i in valid_wp_indices])
            self.waypoints_ddd = []
            for i in range(self.n_waypoints - 1):
                qddd = ((self.waypoints_dd[i + 1] - self.waypoints_dd[i]) / (self.ss_waypoints[i + 1] - self.ss_waypoints[i]))
                self.waypoints_ddd.append(qddd)
            self.waypoints_ddd = np.array(self.waypoints_ddd)

            # Fill the coefficient matrix for scipy.PPoly class
            pp_coeffs = np.zeros((4, self.n_waypoints - 1, self.dof))
            for idof in range(self.dof):
                for iseg in range(self.n_waypoints - 1):
                    pp_coeffs[:, iseg, idof] = [self.waypoints_ddd[iseg, idof] / 6,
                                                self.waypoints_dd[iseg, idof] / 2,
                                                self.waypoints_d[iseg, idof],
                                                self.waypoints[iseg, idof]]
            self.ppoly = PPoly(pp_coeffs, self.ss_waypoints)

        self.ppoly_d = self.ppoly.derivative()
        self.ppoly_dd = self.ppoly.derivative(2)

    def get_duration(self):
        return self._duration

    def eval(self, ss_sam):
        return self.ppoly(ss_sam)

    def evald(self, ss_sam):
        return self.ppoly_d(ss_sam)

    def evaldd(self, ss_sam):
        return self.ppoly_dd(ss_sam)


class SplineInterpolator(Interpolator):
    """Interpolate the given waypoints by cubic spline.

    This interpolator is implemented as a simple wrapper over scipy's
    CubicSpline class.

    Parameters
    ----------
    ss_waypoints: array
        Shaped (N+1,). Path positions of the waypoints.
    waypoints: array
        Shaped (N+1, dof). Waypoints.
    bc_type: str, optional
        Boundary condition. Can be 'not-a-knot', 'clamped', 'natural' or 'periodic'.
        See scipy.CubicSpline documentation for more details.

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
    def __init__(self, ss_waypoints, waypoints, bc_type='clamped'):
        super(SplineInterpolator, self).__init__()
        assert ss_waypoints[0] == 0, "First index must equals zero."
        self.ss_waypoints = np.array(ss_waypoints)
        self.waypoints = np.array(waypoints)
        if np.isscalar(waypoints[0]):
            self.dof = 1
        else:
            self.dof = self.waypoints[0].shape[0]
        self.duration = ss_waypoints[-1]
        assert self.ss_waypoints.shape[0] == self.waypoints.shape[0]
        self.s_start = self.ss_waypoints[0]
        self.s_end = self.ss_waypoints[-1]

        if len(ss_waypoints) == 1:
            def f1(s):
                try:
                    ret = np.zeros((len(s), self.dof))
                    ret[:, :] = self.waypoints[0]
                except TypeError:
                    ret = self.waypoints[0]
                return ret
            def f2(s):
                try:
                    ret = np.zeros((len(s), self.dof))
                except TypeError:
                    ret = np.zeros(self.dof)
                return ret

            self.cspl = f1
            self.cspld = f2
            self.cspldd = f2
        else:
            self.cspl = CubicSpline(ss_waypoints, waypoints, bc_type=bc_type)
            self.cspld = self.cspl.derivative()
            self.cspldd = self.cspld.derivative()

    def get_duration(self):
        return self.duration * self.scaling

    def eval(self, ss_sam):
        return self.cspl(ss_sam / self.scaling)

    def evald(self, ss_sam):
        return self.cspld(ss_sam / self.scaling) / self.scaling

    def evaldd(self, ss_sam):
        # See the `set_scaling` method of `Interpolator` for more
        # details on scaling.
        return self.cspldd(ss_sam / self.scaling) / self.scaling ** 2

    def compute_rave_trajectory(self, robot):
        """ Compute an OpenRAVE trajectory equivalent to this trajectory.

        Parameters
        ----------
        robot: OpenRAVE.Robot

        Returns
        -------
        trajectory: OpenRAVE.Trajectory
        """

        traj = orpy.RaveCreateTrajectory(robot.GetEnv(), "")
        spec = robot.GetActiveConfigurationSpecification('cubic')
        spec.AddDerivativeGroups(1, False)
        spec.AddDerivativeGroups(2, True)

        traj.Init(spec)
        deltas = [0]
        for i in range(len(self.ss_waypoints) - 1):
            deltas.append(self.ss_waypoints[i + 1] - self.ss_waypoints[i])
        if len(self.ss_waypoints) == 1:
            q = self.eval(0)
            qd = self.evald(0)
            qdd = self.evaldd(0)
            traj.Insert(traj.GetNumWaypoints(), list(q) + list(qd) + list(qdd) + [0])
        else:
            qs = self.eval(self.ss_waypoints)
            qds = self.evald(self.ss_waypoints)
            qdds = self.evaldd(self.ss_waypoints)
            for (q, qd, qdd, dt) in zip(qs, qds, qdds, deltas):
                traj.Insert(traj.GetNumWaypoints(),
                            q.tolist() + qd.tolist() + qdd.tolist() + [dt])
        return traj


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

    def get_duration(self):
        return self.duration

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
        self.duration = s_end - s_start
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

    def get_duration(self):
        return self.duration

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
