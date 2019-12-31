"""This module implements interpolators for representing geometric paths and trajectories.
"""
import logging
import numpy as np
from scipy.interpolate import UnivariateSpline, CubicSpline, PPoly
from toppra.utils import deprecated
from toppra.constants import FOUND_OPENRAVE

logger = logging.getLogger(__name__)

if FOUND_OPENRAVE:
    import openravepy as orpy  # pylint: disable=import-error


def propose_gridpoints(path, max_err_threshold=1e-4, max_iteration=100, max_seg_length=0.05):
    """Generate a set of grid pooint for the given path.

    This function operates in multiple passes through the geometric
    path from the start to the end point. In each pass, for each
    segment, the maximum interpolation error is estimated using the
    following equation:

        0.5 * max(abs(p'' * d_segment ^ 2))

    Here p'' is the second derivative of the path and d_segment is the
    length of the segment. Intuitively, at positions with higher
    curvature, there must be more points in order to improve
    approximation quality.

    Arguments
    ---------
    path: Input geometric path.
    max_err_threshold: Maximum worstcase error thrshold allowable.
    max_iteration: Maximum number of iterations.
    max_seg_length: All segments length should be smaller than this value.

    """
    gridpoints_ept = [path.path_interval[0], path.path_interval[1]]
    for iteration in range(max_iteration):
        add_new_points = False
        for idx in range(len(gridpoints_ept) - 1):
            gp_mid = 0.5 * (gridpoints_ept[idx] + gridpoints_ept[idx + 1])
            if (gridpoints_ept[idx + 1] - gridpoints_ept[idx]) > max_seg_length:
                gridpoints_ept.append(gp_mid)
                add_new_points = True
                continue

            dist = gridpoints_ept[idx + 1] - gridpoints_ept[idx]
            max_err = np.max(np.abs(0.5 * path(gp_mid, 2) * dist ** 2))
            if max_err > max_err_threshold:
                add_new_points = True
                gridpoints_ept.append(gp_mid)
        gridpoints_ept = sorted(gridpoints_ept)
        if not add_new_points:
            break
    if iteration == max_iteration - 1:
        raise ValueError("Unable to find a good gridpoint for this path.")
    return gridpoints_ept


class AbstractGeometricPath(object):
    """Base geometric path.

    Derive geometric paths classes should derive the below abstract methods.
    """

    def __call__(self, path_positions, order=0):
        # type: (np.ndarray, int) -> np.ndarray
        """Evaluate the path at given positions.

        Parameters
        ----------
            path_positions: float or np.ndarray
                Path positions to evaluate the interpolator.
            order: int
                Order of the evaluation call.

                - 0: position
                - 1: first-order derivative
                - 2: second-order derivative

        Returns
        -------
            np.ndarray
                Evaluated values.

        """
        raise NotImplementedError

    @property
    def dof(self):
        """Return the degrees-of-freedom of the path."""
        raise NotImplementedError

    @property
    def path_interval(self):
        """Return the starting and ending path positions.

        Returns
        -------
        out:
            The starting and ending path positions.

        """
        raise NotImplementedError


class RaveTrajectoryWrapper(AbstractGeometricPath):
    """An interpolator that wraps OpenRAVE's :class:`GenericTrajectory`.

    Only trajectories using quadratic interpolation or cubic
    interpolation are supported.  The trajectory is represented as a
    piecewise polynomial. The polynomial could be quadratic or cubic
    depending the interpolation method used by the input trajectory
    object.

    """
    @staticmethod
    def _extract_interpolation_method(spec):
        _interpolation = spec.GetGroupFromName('joint').interpolation
        if _interpolation not in ['quadratic', 'cubic']:
            raise ValueError(
                "This class only handles trajectories with quadratic or cubic interpolation"
            )
        return _interpolation

    def __init__(self, traj, robot):
        # type: (orpy.RaveTrajectory, orpy.Robot) -> None
        """Initialize the Trajectory Wrapper.

        Parameters
        ----------
        traj:
            An OpenRAVE joint trajectory.
        robot:
            An OpenRAVE robot.

        """
        super(RaveTrajectoryWrapper, self).__init__()
        self.traj = traj  #: init
        spec = traj.GetConfigurationSpecification()
        self._dof = robot.GetActiveDOF()
        self._interpolation = self._extract_interpolation_method(spec)
        self._duration = traj.GetDuration()

        all_waypoints = traj.GetWaypoints(0, traj.GetNumWaypoints()).reshape(
            traj.GetNumWaypoints(), -1)
        valid_wp_indices = [0]
        self.ss_waypoints = [0.0]
        for i in range(1, traj.GetNumWaypoints()):
            dt = spec.ExtractDeltaTime(all_waypoints[i])
            if dt > 1e-5:  # If delta is too small, skip it.
                valid_wp_indices.append(i)
                self.ss_waypoints.append(self.ss_waypoints[-1] + dt)
        self.ss_waypoints = np.array(self.ss_waypoints)
        n_waypoints = len(valid_wp_indices)

        def _extract_waypoints(order):
            return np.array([
                spec.ExtractJointValues(all_waypoints[i], robot, robot.GetActiveDOFIndices(), order)
                for i in valid_wp_indices
            ])

        def _make_ppoly():
            if n_waypoints == 1:
                waypoints = _extract_waypoints(0)
                pp_coeffs = np.zeros((1, 1, self.dof))
                for idof in range(self.dof):
                    pp_coeffs[0, 0, idof] = waypoints[0, idof]
                return PPoly(pp_coeffs, [0, 1])

            if self._interpolation == "quadratic":
                waypoints = _extract_waypoints(0)
                waypoints_d = _extract_waypoints(1)
                waypoints_dd = []
                for i in range(n_waypoints - 1):
                    qdd = ((waypoints_d[i + 1] - waypoints_d[i]) /
                           (self.ss_waypoints[i + 1] - self.ss_waypoints[i]))
                    waypoints_dd.append(qdd)
                waypoints_dd = np.array(waypoints_dd)

                # Fill the coefficient matrix for scipy.PPoly class
                pp_coeffs = np.zeros((3, n_waypoints - 1, self.dof))
                for idof in range(self.dof):
                    for iseg in range(n_waypoints - 1):
                        pp_coeffs[:, iseg, idof] = [
                            waypoints_dd[iseg, idof] / 2,
                            waypoints_d[iseg, idof],
                            waypoints[iseg, idof]
                        ]
                return PPoly(pp_coeffs, self.ss_waypoints)

            if self._interpolation == "cubic":
                waypoints = _extract_waypoints(0)
                waypoints_d = _extract_waypoints(1)
                waypoints_dd = _extract_waypoints(2)
                waypoints_ddd = []
                for i in range(n_waypoints - 1):
                    qddd = ((waypoints_dd[i + 1] - waypoints_dd[i]) /
                            (self.ss_waypoints[i + 1] - self.ss_waypoints[i]))
                    waypoints_ddd.append(qddd)
                waypoints_ddd = np.array(waypoints_ddd)

                # Fill the coefficient matrix for scipy.PPoly class
                pp_coeffs = np.zeros((4, n_waypoints - 1, self.dof))
                for idof in range(self.dof):
                    for iseg in range(n_waypoints - 1):
                        pp_coeffs[:, iseg, idof] = [
                            waypoints_ddd[iseg, idof] / 6,
                            waypoints_dd[iseg, idof] / 2,
                            waypoints_d[iseg, idof],
                            waypoints[iseg, idof]
                        ]
                return PPoly(pp_coeffs, self.ss_waypoints)
            raise ValueError("An error has occured. Unable to form PPoly.")

        self.ppoly = _make_ppoly()

    @deprecated
    def get_duration(self):
        """Return the path's duration."""
        return self.duration

    @deprecated
    def get_dof(self):  # type: () -> int
        """Return the path's dof."""
        return self.dof

    @property
    def duration(self):
        """Return the duration of the path."""
        return self._duration

    @property
    def path_interval(self):
        """Return the start and end points."""
        return np.array([0, self._duration])

    @property
    def dof(self):
        return self._dof

    def __call__(self, ss_sam, order=0):
        if order == 0:
            return self.eval(ss_sam)
        if order == 1:
            return self.evald(ss_sam)
        if order == 2:
            return self.evaldd(ss_sam)
        raise ValueError("Order must be 0, 1 or 2.")

    def eval(self, ss_sam):
        """Evalute path postition."""
        return self.ppoly(ss_sam)

    def evald(self, ss_sam):
        """Evalute path velocity."""
        return self.ppoly.derivative()(ss_sam)

    def evaldd(self, ss_sam):
        """Evalute path acceleration."""
        return self.ppoly.derivative(2)(ss_sam)


class SplineInterpolator(AbstractGeometricPath):
    """Interpolate the given waypoints by cubic spline.

    This interpolator is implemented as a simple wrapper over scipy's
    CubicSpline class.

    Parameters
    ----------
    ss_waypoints: array
        Shaped (N+1,). Path positions of the waypoints.
    waypoints: array
        Shaped (N+1, dof). Waypoints.
    bc_type: optional
        Boundary conditions of the spline. Can be 'not-a-knot',
        'clamped', 'natural' or 'periodic'.

        - 'not-a-knot': The most default option, return to natural
          looking spline.
        - 'clamped': First-order derivatives of the spline at the two
          end are zeroed.

        See scipy.CubicSpline documentation for more details.

    Attributes
    ----------
    dof : int
        Output dimension of the function
    cspl : :class:`scipy.interpolate.CubicSpline`
        The underlying cubic spline.
    """

    def __init__(self, ss_waypoints, waypoints, bc_type='not-a-knot'):
        super(SplineInterpolator, self).__init__()
        self.ss_waypoints = np.array(ss_waypoints)  # type: np.ndarray
        self.waypoints = np.array(waypoints)  # type: np.ndarray
        assert self.ss_waypoints.shape[0] == self.waypoints.shape[0]

        if len(ss_waypoints) == 1:

            def _1dof_cspl(s):
                try:
                    ret = np.zeros((len(s), self.dof))
                    ret[:, :] = self.waypoints[0]
                except TypeError:
                    ret = self.waypoints[0]
                return ret

            def _1dof_cspld(s):
                try:
                    ret = np.zeros((len(s), self.dof))
                except TypeError:
                    ret = np.zeros(self.dof)
                return ret

            self.cspl = _1dof_cspl
            self.cspld = _1dof_cspld
            self.cspldd = _1dof_cspld
        else:
            self.cspl = CubicSpline(ss_waypoints, waypoints, bc_type=bc_type)
            self.cspld = self.cspl.derivative()
            self.cspldd = self.cspld.derivative()

    def __call__(self, path_positions, order=0):
        if order == 0:
            return self.cspl(path_positions)
        if order == 1:
            return self.cspld(path_positions)
        if order == 2:
            return self.cspldd(path_positions)
        raise ValueError("Invalid order %s" % order)

    def get_waypoints(self):
        """Return the appropriate scaled waypoints."""
        return self.ss_waypoints, self.waypoints

    @deprecated
    def get_duration(self):
        """Return the path's duration."""
        return self.duration

    @property
    def duration(self):
        """Return the duration of the path."""
        return self.ss_waypoints[-1] - self.ss_waypoints[0]

    @property
    def path_interval(self):
        """Return the start and end points."""
        return np.array([self.ss_waypoints[0], self.ss_waypoints[-1]])

    @deprecated
    def get_path_interval(self):
        """Return the path interval."""
        return self.path_interval

    @property
    def dof(self):
        if np.isscalar(self.waypoints[0]):
            return 1
        return self.waypoints[0].shape[0]

    @deprecated
    def get_dof(self):  # type: () -> int
        """Return the path's dof."""
        return self.dof

    @deprecated
    def eval(self, ss_sam):
        """Return the path position."""
        return self.cspl(ss_sam)

    @deprecated
    def evald(self, ss_sam):
        """Return the path velocity."""
        return self.cspld(ss_sam)

    @deprecated
    def evaldd(self, ss_sam):
        """Return the path acceleration."""
        return self.cspldd(ss_sam)

    def compute_rave_trajectory(self, robot):
        """Compute an OpenRAVE trajectory equivalent to this trajectory.

        Parameters
        ----------
        robot:
            Openrave robot.

        Returns
        -------
        trajectory:
            Equivalent openrave trajectory.
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
            q = self(0)
            qd = self(0, 1)
            qdd = self(0, 2)
            traj.Insert(traj.GetNumWaypoints(),
                        list(q) + list(qd) + list(qdd) + [0])
        else:
            qs = self(self.ss_waypoints)
            qds = self(self.ss_waypoints, 1)
            qdds = self(self.ss_waypoints, 2)
            for (q, qd, qdd, dt) in zip(qs, qds, qdds, deltas):
                traj.Insert(traj.GetNumWaypoints(),
                            q.tolist() + qd.tolist() + qdd.tolist() + [dt])
        return traj


class UnivariateSplineInterpolator(AbstractGeometricPath):
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
            self._dof = 1
        else:
            self._dof = waypoints[0].shape[0]
        assert self.ss_waypoints.shape[0] == self.waypoints.shape[0]
        self.uspl = []
        for i in range(self.dof):
            self.uspl.append(
                UnivariateSpline(self.ss_waypoints, self.waypoints[:, i]))
        self.uspld = [spl.derivative() for spl in self.uspl]
        self.uspldd = [spl.derivative() for spl in self.uspld]

    @property
    def dof(self):
        return self._dof

    @property
    def path_interval(self):
        """Return the path interval."""
        return [self.ss_waypoints[0], self.ss_waypoints[-1]]

    def __call__(self, ss_sam, order=0):
        data = []
        if order == 0:
            for spl in self.uspl:
                data.append(spl(ss_sam))
        elif order == 1:
            for spl in self.uspld:
                data.append(spl(ss_sam))
        elif order == 2:
            for spl in self.uspldd:
                data.append(spl(ss_sam))
        return np.array(data).T

    def eval(self, ss_sam):
        """Return the path position."""
        data = []
        for spl in self.uspl:
            data.append(spl(ss_sam))
        return np.array(data).T

    def evald(self, ss_sam):
        """Return the path velocity."""
        data = []
        for spl in self.uspld:
            data.append(spl(ss_sam))
        return np.array(data).T

    def evaldd(self, ss_sam):
        """Return the path acceleration."""
        data = []
        for spl in self.uspldd:
            data.append(spl(ss_sam))
        return np.array(data).T


class PolynomialPath(AbstractGeometricPath):
    """ A class representing polynominal paths.

    If coeff is a 1d array, the polynomial's equation is given by

    .. math::

    coeff[0] + coeff[1] s + coeff[2] s^2 + ...

    If coeff is a 2d array, the i-th joint position is the polynomial

    .. math::

    coeff[i, 0] + coeff[i, 1] s + coeff[i, 2] s^2 + ...
    """

    def __init__(self, coeff, s_start=0.0, s_end=1.0):
        # type: (np.ndarray, float, float) -> None
        """Initialize the polynomial path.

        Parameters
        ----------
        coeff
            Coefficients of the polynomials.
        s_start
            Starting path position.
        s_end
            Ending path position.
        """
        super(PolynomialPath, self).__init__()
        self.coeff = np.array(coeff)
        self.s_end = s_end
        self.s_start = s_start
        if np.isscalar(self.coeff[0]):
            self.poly = [np.polynomial.Polynomial(self.coeff)]
            self.coeff = self.coeff.reshape(1, -1)
        else:
            self.poly = [
                np.polynomial.Polynomial(self.coeff[i])
                for i in range(self.dof)
            ]

        self.polyd = [poly.deriv() for poly in self.poly]
        self.polydd = [poly.deriv() for poly in self.polyd]

    def __call__(self, path_positions, order=0):
        if order == 0:
            return self.eval(path_positions)
        if order == 1:
            return self.evald(path_positions)
        if order == 2:
            return self.evaldd(path_positions)
        raise ValueError("Invalid order %s" % order)

    @property
    def dof(self):
        return self.coeff.shape[0]

    @property
    def duration(self):
        """Return the duration of the path."""
        return self.s_end - self.s_start

    @property
    def path_interval(self):
        return np.array([self.s_start, self.s_end])

    @deprecated
    def get_path_interval(self):
        """Return the path interval."""
        return self.path_interval

    @deprecated
    def get_duration(self):
        """Return the path duration."""
        return self.duration

    @deprecated
    def get_dof(self):
        """Return the path's dof."""
        return self.dof

    def eval(self, ss_sam):
        """Return the path position."""
        res = [poly(np.array(ss_sam)) for poly in self.poly]
        if self.dof == 1:
            return np.array(res).flatten()
        return np.array(res).T

    def evald(self, ss_sam):
        """Return the path velocity."""
        res = [poly(np.array(ss_sam)) for poly in self.polyd]
        if self.dof == 1:
            return np.array(res).flatten()
        return np.array(res).T

    def evaldd(self, ss_sam):
        """Return the path acceleration."""
        res = [poly(np.array(ss_sam)) for poly in self.polydd]
        if self.dof == 1:
            return np.array(res).flatten()
        return np.array(res).T
