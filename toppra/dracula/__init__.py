"""
TOPP-RA Dracula Interface.

Wrapper for toppra that takes in waypoints, v and a limits.
For rapid splining from Drake.
"""

import datetime
import logging
import os

import numpy as np
from scipy.interpolate import CubicSpline

import toppra as ta
import toppra.algorithm as algo
from toppra import constraint  # , interpolator  # needed by custom gridpoints

from .zero_acceleration_start_end import impose_natural_bc

ta.setup_logging("INFO")
# min epsilon for treating two angles the same, positive float
JNT_DIST_EPS = 2e-3  # nominally defined as L2 norm in joint space, i.e. in rad
# toppra does not respect velocity limit precisely
V_LIM_EPS = 0.12
A_LIM_EPS = 0.07
# https://frankaemika.github.io/docs/control_parameters.html#constants
V_MAX = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100])
A_MAX = np.array([15, 7.5, 10, 12.5, 15, 20, 20])
DATA_DIR = "/data/toppra/input_data"
os.makedirs(DATA_DIR, exist_ok=True)
log_t0 = datetime.datetime.now(datetime.timezone.utc).strftime(
    r"%Y%m%dT%H%M%S%z"
)
log_path = f"/data/toppra/logs/{log_t0}.log"
os.makedirs(os.path.dirname(log_path), exist_ok=True)
logger = logging.getLogger("toppra")
open(log_path, "a").close()  # create log file if it does not exist
fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)


def _dump_input_data(**kwargs):
    """Dump input data for debugging when certain env vars are detected.

    t is the timestamp of the input data, used as the filename.
    Variables must be passed as named kwargs.
    """
    t = datetime.datetime.now(datetime.timezone.utc).strftime(
        r"%Y%m%dT%H%M%S.%f%z"
    )
    path = os.path.join(DATA_DIR, f"{t}.npz")
    np.savez(path, **kwargs)
    logger.info(f"Debug environment detected, input data saved to: {path}")


def _check_waypts(waypts, vlim):
    """
    Perform two checks on the input waypoints.

    Return minimum distance between adjacent pairs for checking duplications,
    and the estimated path length (run time) of the trajectory given the
    velocity limit and the waypoints in joint space.
    """
    pair_dist = np.diff(waypts, axis=0)  # (N-1, N_dof)
    pair_t = np.abs(pair_dist) / vlim[:, 1]
    t_sum = pair_t.max(axis=1).sum()
    min_pair_dist = np.linalg.norm(pair_dist, axis=1).min()
    return min_pair_dist, t_sum


class DraculaToppra:
    """Class for an optimisation session. Use the functional wrappers."""

    def __init__(self, waypts, vlim, alim):
        """Initialise with session data and perform common initial prep."""
        if any(map(os.getenv, ["SIM_ROBOT", "TOPPRA_DEBUG"])):
            _dump_input_data(waypts=waypts, vlim=vlim, alim=alim)
        # check vlim and alim are sufficiently large, at least eps
        assert (np.abs(vlim.flatten()) > V_LIM_EPS).all(), (
            f"vlim magnitude must be larger than V_LIM_EPS = {V_LIM_EPS}:"
            f"\n{vlim}"
        )
        assert (np.abs(alim.flatten()) > A_LIM_EPS).all(), (
            f"vlim magnitude must be larger than V_LIM_EPS = {A_LIM_EPS}:"
            f"\n{vlim}"
        )
        # check for duplicates
        self.min_pair_dist, t_sum = _check_waypts(waypts, vlim)
        if self.min_pair_dist < JNT_DIST_EPS:  # issue a warning and try anyway
            logger.warning(
                "Duplicates found in input waypoints. This is not recommended,"
                " especially for the beginning and the end of the trajectory. "
                "Toppra might throw a controllability exception. "
                "Attempting to optimise trajectory anyway..."
            )
        self.waypts = waypts
        self.vlim = vlim
        self.alim = alim
        if t_sum < 0.5:
            t_sum = 0.5  # 0.15 seems to be the minimum path length required
        # initial x for toppra's path, essentially normalised time on x axis
        # rescale by given speed limits.
        # only applies to ParametrizeSpline.
        self.path_length_limit = 100 * t_sum  # empirical magic number
        # magic number 0.3 because despite the fact that t_sum is the minimum
        # time required to visit all given waypoints, toppra needs a smaller
        # number for controllabiility
        # it will find that the needed total path length > t_sum in the end
        x = np.linspace(0, 0.3 * t_sum, waypts.shape[0])
        # specifying natural here doensn't make a difference
        # toppra only produces clamped cubic splines
        self.path = ta.SplineInterpolator(x, waypts.copy(), bc_type="clamped")
        # avoid going over limit taking into account toppra's precision
        self.pc_vel = constraint.JointVelocityConstraint(
            vlim - np.sign(vlim) * V_LIM_EPS
        )
        self.alim_coeffs = np.ones(alim.shape[0])  # IV for multiplication

    def lims_obeyed(self, traj, raise_2nd_order):
        """
        Verify that velocity and acceleration constraints are strictly obeyed.

        Either return None (all limits obeyed) or an array of coefficients
        for alims.
        """
        try:
            x = traj.cspl.x
        except AttributeError:
            x = traj._ts
        for order, lim in enumerate([self.vlim, self.alim], start=1):
            deriv = traj(x, order=order)
            # get mask not satisfying both uppwer and lower lims
            i, j = np.where(~((lim[:, 0] < deriv) & (deriv < lim[:, 1])))
            if i.size or j.size:  # should be same size
                # only do the following calculation if violation is found
                signed_lim = np.where(deriv > 0, lim[:, 1], lim[:, 0])
                # exccess on both sides are +ve, under-limit entries are -ve
                # only [i, j] entries are +ve for both excess variables below
                excess = np.sign(deriv) * (deriv - signed_lim)
                excess_percent = excess / np.abs(signed_lim)
                if order == 1 or raise_2nd_order:
                    # unexpected, toppra should respect constraints up to eps
                    logger.error(
                        f"Dynamic constraint violated: order: {order}\n"
                        f"derivative:\n{np.around(deriv[i, j], 4)}\n"
                        f"limits:\n{np.around(signed_lim[i, j], 4)}\n"
                        f"excess:\n{np.around(excess[i, j], 4)}\n"
                        "excees_percent:\n"
                        f"{np.around(excess_percent[i, j], 4)}"
                    )
                    raise ValueError("Epsilon needs to be tweaked.")
                # order == 2 and parametrizer == "Spline", overshooot expected
                # return a vector of reduction coefficients
                excess_percent_joint = excess_percent.max(axis=0)
                excess_percent_joint[excess_percent_joint <= 0] = 0
                # set minimum reduction step, or sometimes it goes too slowly
                excess_percent_joint[excess_percent_joint < 0.02] = 0.02
                # slower than 1 - percent
                self.alim_coeffs *= 1 / (1 + excess_percent_joint)
                return False
        return True

    def compute_spline_varying_alim(self):
        """Compute spline-based jnt_traj one-pass using current alim."""
        # Can be either Collocation (0) or Interpolation (1).
        # Interpolation gives more accurate results with
        # slightly higher computational cost
        pc_acc = constraint.JointAccelerationConstraint(
            self.alim_coeffs.reshape(-1, 1) * self.alim
            - np.sign(self.alim) * A_LIM_EPS,
            discretization_scheme=constraint.DiscretizationType.Interpolation,
        )
        # Use the default gridpoints=None to let
        # interpolator.propose_gridpoints calculate gridpoints
        # that sufficiently covers the path.
        # this ensures the instance is controllable and avoids error:
        #     "An error occurred when computing controllable velocities.
        #     The path is not controllable, or is badly conditioned.
        #     Error: Instance is not controllable"
        # If using clamped as boundary condition, the default gridpoints error
        # 1e-3 is OK and we don't need to calculate gridpoints.
        # Boundary condition "natural" especially needs support by
        # smaller error.
        instance = algo.TOPPRA(
            [self.pc_vel, pc_acc],
            self.path,
            solver_wrapper="seidel",
            parametrizer="ParametrizeSpline",
        )
        return self._compute_and_check_traj(instance)

    def compute_const_accel(self):
        """Compute optimised trajectory for ParametrizeConstAccel."""
        pc_acc = constraint.JointAccelerationConstraint(
            self.alim - np.sign(self.alim) * A_LIM_EPS,
            discretization_scheme=constraint.DiscretizationType.Interpolation,
        )
        instance = algo.TOPPRA(
            [self.pc_vel, pc_acc],
            self.path,
            solver_wrapper="seidel",
            parametrizer="ParametrizeConstAccel",
            gridpt_min_nb_points=1000,  # ensure eps ~ O(1e-2)
        )
        return self._compute_and_check_traj(instance)

    def _compute_and_check_traj(self, instance):
        traj = instance.compute_trajectory(0, 0)
        if traj is None:  # toppra has failed
            if (
                self.min_pair_dist < JNT_DIST_EPS
            ):  # duplicates are probably why
                logger.error(
                    "Duplicates not allowed in input waypoints. "
                    "At least one pair of adjacent waypoints have "
                    f"distance less than epsilon = {JNT_DIST_EPS}."
                )
            logger.error(
                f"Failed waypts:\n{self.waypts}"
                f"\nvlim:\n{self.vlim}\nalim:\n{self.alim}"
            )
            raise RuntimeError("Toppra failed to compute trajectory.")
        return traj

    def truncate_traj(self, traj, parametrizer):
        """Finish CubidSpline after it's settled and ready for return.

        Take care of ending truncation and natural boundary condition.
        """
        # If the initial estimated path length (run time) of the trajectory
        # isn't very close to the actual computed one,
        # (say off by a factor of 2),
        # toppra may go a bit crazy sometimes extends the spline to x >~ 1e3.
        # This cannot be fixed by tweaking the fit.
        # N_samples = 39 may have this behaviour unless x_max > 1.6,
        # while N_samples = 46 may require x_max < 1.6 to be controllable.
        # So we keep x_max towards the small side to guarantee controllability.
        # We check that and truncate the spline at a predefined upper bound.
        if parametrizer == "ParametrizeSpline":
            x = traj.cspl.x
        elif parametrizer == "ParametrizeConstAccel":
            x = traj._ts
        else:
            raise ValueError(f"Invalid parametrizer type: {parametrizer}")

        if x[-1] <= self.path_length_limit:  # quick and dirty detection
            return traj  # should be all good, nothing to do here

        if parametrizer == "ParametrizeSpline":
            mask = x <= self.path_length_limit  # the good ones on the left
            logger.warning(
                "Suboptimal trajectory derived, input waypoints likely "
                "ill-formed. "
                f"x > {self.path_length_limit:.3f} for "
                f"{(~mask).sum()}/{x.size} ending knots, "
                f"x_max: {x[-1]:.3f}."
            )
            # now truncate and check that it is still close to the original end
            cs = CubicSpline(x[mask], traj.cspl(x[mask]), bc_type="natural")
            traj.cspl = cs  # just overwrite attr
            new_end_pos = cs(cs.x[-1])
        else:  # parametrizer == ParametrizeConstAccel
            # look from right for the first True or the last continuous Falses
            mask = (
                np.linalg.norm(np.abs((traj(x) - self.waypts[-1])), axis=1)
                > JNT_DIST_EPS
            )
            # keep only the first False from the ending False block.
            traj._ts = traj._ts[mask.size - np.argmax(np.flip(mask)) + 1]
            new_end_pos = traj(traj.duration)
        assert np.linalg.norm(new_end_pos - self.waypts[-1]) < JNT_DIST_EPS, (
            f"Truncated trajectory, ending at\n{new_end_pos},\n"
            f"no longer arrives at the original ending waypoint\n"
            f"{self.waypts[-1]}\n"
            f"given JNT_DIST_EPS = {JNT_DIST_EPS}. "
            "Try closer and smoother waypoints with a smaller path length "
            "if using ParametrizeSpline, and try a higher cmd_rate "
            "if using ParametrizeConstAccel."
        )
        logger.info(
            f"Time-optimised trajectory truncated at limit x = "
            f"{self.path_length_limit:.3f}, still arriving at the "
            f"original ending waypoint up to JNT_DIST_EPS: {JNT_DIST_EPS}. "
            f"Duration after truncation: {traj.duration:.3f}."
        )
        return traj


def run_topp_spline(waypts, vlim, alim, verify_lims=True, return_cs=False):
    """Call toppra obeying velocity and acceleration limits and naturalness.

    Use of run_topp_const_accel() or run_topp_jnt_crt() is encouraged for
    faster/smoother motion.

    No consecutive duplicates in the waypoints allowed.
    Return natural cubic spline coefficients by default.
    Required args:
        waypts          ndarray, (N, dof)
        vlim            ndarray, (dof, 2)
        alim            ndarray, (dof, 2)
    Optional args:
        verify_lims     bool
        return_cs       cs
    """
    topp = DraculaToppra(waypts, vlim, alim)
    traj = topp.compute_spline_varying_alim()
    if verify_lims:  # flag for checking if vlim is obeyed
        logger.info("Verifying that given limits are strictly obeyed...")
        while not topp.lims_obeyed(traj, raise_2nd_order=False):
            logger.info(
                "Current iteration violates acceleration limits, trying "
                f"new coefficients: {repr(np.around(topp.alim_coeffs, 3))}..."
            )
            traj = topp.compute_spline_varying_alim()
    traj = topp.truncate_traj(traj, parametrizer="ParametrizeSpline")
    # Toppra goes a bit wider than a precise natural cubic spline.
    # We could find the leftmost and rightmost common roots of all dof,
    # which are the original end points, but that algorithm not guaranteed
    # to converge below a sufficiently small error and is not efficient.
    # We could also use scipy to respline naturally, but it only support
    # one boundary condition, either zero velocity or zero accleration.
    # Manually treat two ends to force both derivatives to be zero.
    cs = impose_natural_bc(traj.cspl)
    logger.info(
        f"Finished computing time-optimised cubic spline trajectory of "
        f"{cs.x.size} knots, duration: {traj.duration:.3f} s. "
    )
    logger.warning(
        "To preserve constraints, continuity, and boundary conditions, this "
        "computed CubicSpline polynomial MUST NOT be resplined arbitrarily."
    )
    if return_cs:
        return cs
    return (
        cs.x.size,
        np.ascontiguousarray(cs.x, dtype=np.float64),
        np.ascontiguousarray(cs.c, dtype=np.float64),
    )


def run_topp_const_accel(waypts, vlim, alim, cmd_rate=1000, verify_lims=True):
    """Use non-spline based parameteriser and get all raw samples directly.

    We cannot spline the result from ParametrizerConstAccel as that would
    violate the constraints. Must evaluate on the result directly.
    Command rate is in Hz.
    """
    topp = DraculaToppra(waypts, vlim, alim)
    traj = topp.compute_const_accel()
    if verify_lims:
        logger.info("Verifying that given limits are strictly obeyed...")
        topp.lims_obeyed(traj, raise_2nd_order=True)
    traj = topp.truncate_traj(traj, parametrizer="ParametrizeConstAccel")

    t = np.arange(0, traj.duration, 1 / cmd_rate)  # duration is cut short
    jnt_pos = traj(t)
    assert np.linalg.norm(jnt_pos[[-1]] - waypts[-1]) < JNT_DIST_EPS, (
        f"Time-optimised raw trajectory, ending at\n{jnt_pos[-1]},\n"
        f"no longer arrives at the original ending waypoint\n"
        f"{waypts[-1]}\n"
        f"given JNT_DIST_EPS = {JNT_DIST_EPS}, usually because it is unable "
        "to sufficiently cover the full duration. Try a higher command rate."
    )
    logger.info(
        f"Finished computing time-optimised raw trajectory of "
        f"{t.size} samples, duration: {traj.duration:.4f} -> {t[-1]:.4f} s. "
    )
    return t.size, t, jnt_pos


def _find_waypts_indices(waypts, cs):
    """Find the indices for the original waypoints in cubic spline knots."""
    idx = np.zeros(waypts.shape[0], dtype=int)
    k = 0  # index for knots, scan all knots left to right, start at the 0th
    for i, waypt in enumerate(waypts):
        waypt_min_err = float("inf")  # always reset error for current waypt
        while k < cs.x.size:
            err = np.linalg.norm(cs(cs.x[k]) - waypt)
            if err <= waypt_min_err + (i > 0) * JNT_DIST_EPS:
                # for non-initial point, error may fluctuate up to eps
                # only call it farther if it's farther than min + eps
                waypt_min_err = err
            else:  # we've found the closest knot at the previous knot, k-1
                idx[i] = k - 1
                break
            k += 1
        idx[i] = k - 1
    assert idx[0] == 0, "The first knot is not the beginning waypoint"
    assert all(
        idx[1:] != 0
    ), "Failed to find all original waypoints in CubicSpline"
    assert idx[-1] == cs.x.size - 1, "The last knot is not the ending waypoint"
    return idx


def run_topp_jnt_crt(
    waypts_jnt,  # (N, Ndof)
    vlim_jnt,  # (Ndof, 2)
    alim_jnt,  # (Ndof, 2)
    waypts_crt,  # (N, 3)
    vlim_crt,  # (3, 2)
    alim_crt,  # (3, 2)
    return_cs=False,
):
    """Optimise joint-space trajectory with additional cartesian limits.

    Only spline parameteriser is supported as this module has no access to IK,
    and the Cartesian limits are incorporated by matching the waypoints.
    """
    logger.info("Optimising joint-space trajectory...")
    cs_jnt = run_topp_spline(
        waypts_jnt,
        vlim_jnt,
        alim_jnt,
        verify_lims=True,
        return_cs=True,
    )
    logger.info("Optimising Cartesian trajectory...")
    cs_crt = run_topp_spline(
        waypts_crt,
        vlim_crt,
        alim_crt,
        verify_lims=True,
        return_cs=True,
    )
    # find new indices for original waypts_jnt in cs_jnt
    idx_jnt = _find_waypts_indices(waypts_jnt, cs_jnt)
    # find new indices for original waypts_crt in cs_crt
    idx_crt = _find_waypts_indices(waypts_crt, cs_crt)
    # now modify timing of cs_jnt to take into account cartesian optimisation
    # starting from the 1st waypoint (after the 0th)
    x_jnt_new = cs_jnt.x.copy()
    for i, (m, n) in enumerate(zip(idx_jnt[1:], idx_crt[1:]), start=1):
        dx0 = cs_jnt.x[m] - cs_jnt.x[idx_jnt[i - 1]]
        dx = cs_crt.x[n] - cs_crt.x[idx_crt[i - 1]]
        if dx > dx0:  # need to slow down for cartesian constraints
            # uniformly dilate x for knots in the current waypoint segment
            x_l = x_jnt_new[idx_jnt[i - 1]]  # x at left waypt
            x_r = x_jnt_new[m]  # x at right waypt
            x = x_jnt_new[idx_jnt[i - 1] + 1 : m + 1]  # x of knots to modify
            x_jnt_new[idx_jnt[i - 1] + 1 : m + 1] = x_l + (x - x_l) * dx / dx0
            # shift x for all future knots after current waypt
            x_jnt_new[m + 1 :] += x_jnt_new[m] - x_r
    cs = CubicSpline(x_jnt_new, cs_jnt(cs_jnt.x), bc_type="clamped")
    impose_natural_bc(cs)
    logger.info(
        f"Finished optimising trajectory of {cs.x.size} knots "
        f"with combined constraints, duration: {cs.x[-1]:.3f} s. "
    )
    logger.warning(
        "To preserve constraints, continuity, and boundary conditions, this "
        "computed CubicSpline polynomial MUST NOT be resplined arbitrarily."
    )
    if return_cs:
        return cs
    return (
        cs.x.size,
        np.ascontiguousarray(cs.x, dtype=np.float64),
        np.ascontiguousarray(cs.c, dtype=np.float64),
    )
