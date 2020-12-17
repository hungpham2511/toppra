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
DIST_EPS = 2e-3  # nominally defined as L2 norm in joint space, i.e. in rad
# toppra does not respect velocity limit precisely
V_LIM_EPS = 0.09
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


def _check_waypoints(waypts, vlim):
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


def _verify_lims(cs, vlim, alim):
    """Verify that velocity and acceleration constraints are strictly obeyed.

    Either return None (all limits obeyed) or a double coeffcient for alims.
    TODO(@dyt): add proper unit tests.
    """
    for order, lim in enumerate([vlim, alim], start=1):
        deriv = cs.derivative(order)(cs.x)
        # get mask not satisfying both uppwer and lower lims
        i, j = np.where(~((lim[:, 0] < deriv) & (deriv < lim[:, 1])))
        if i.size or j.size:  # should be same size
            # only do the following calculation if violation is found
            signed_lim = np.where(deriv > 0, lim[:, 1], lim[:, 0])
            # exccess on both sides are +ve, under-limit entries are -ve
            excess = np.sign(deriv) * (deriv - signed_lim)  # only [i, j] +ve
            excess_percent = excess / np.abs(signed_lim)  # only [i, j] +ve
            if order == 1:  # unexpected, toppra generally respects vlim
                logger.error(
                    f"Dynamic constraint violated: order: {order}\n"
                    f"derivative:\n{np.around(deriv[i, j], 4)}\n"
                    f"limits:\n{np.around(signed_lim[i, j], 4)}\n"
                    f"excess:\n{np.around(excess[i, j], 4)}\n"
                    f"excees_percent:\n{np.around(excess_percent[i, j], 4)}"
                )
                raise ValueError("V_LIM_EPS needs to be tweaked")
            # else order=2, expected, return a vector of reduction coefficients
            excess_percent_joint = excess_percent.max(axis=0)
            excess_percent_joint[excess_percent_joint <= 0] = 0
            # set minimum reduction step, sometimes it goes too slowly
            excess_percent_joint[excess_percent_joint < 0.02] = 0.02
            return 1 / (1 + excess_percent_joint)  # slower than 1-percent
    return None


def run_topp(
    waypts,  # ndarray, (N, dof)
    vlim,  # ndarray, (dof, 2)
    alim,  # ndarray, (dof, 2)
    verify_lims=True,
    return_cs=False,
):
    """Call toppra obeying velocity and acceleration limits and naturalness.

    No consecutive duplicates in the waypoints allowed.
    Return natural cubic spline coefficients by default.
    Set path_length_limit to None to remove any limit on the path length.
    Use with caution as the path length is the time parameter and if taken
    at face value as in unit of seconds, it will result in very lengthy
    trajectories.
    """
    if any(map(os.getenv, ["SIM_ROBOT", "TOPPRA_DEBUG"])):
        _dump_input_data(waypts=waypts, vlim=vlim, alim=alim)
    # check for duplicates, assert adjacent pairs have distance not equal to 0
    min_pair_dist, t_sum = _check_waypoints(waypts, vlim)
    if min_pair_dist < DIST_EPS:  # issue a warning and try anyway
        logger.warning(
            "Duplicates found in input waypoints. This is not recommended, "
            "especially for the beginning and the end of the trajectory. "
            "Toppra might throw a controllability exception. "
            "Attempting to optimise trajectory anyway..."
        )
    if t_sum < 0.5:
        t_sum = 0.5  # 0.15 seems to be the minimum path length required
    # initial x for toppra's path, essentially normalised time on x axis
    # rescale by given speed limits
    path_length_limit = 100 * t_sum  # magic number, cap at 100 x the time
    N_samples = waypts.shape[0]
    # magic number 0.1 because despite the fact that t_sum is the minimum
    # time required to visit all given waypoints, toppra needs a smaller
    # number for controllabiility
    # it will find that the needed total path length > t_sum in the end
    x = np.linspace(0, 0.3 * t_sum, N_samples)
    # specifying natural here doensn't make a difference
    # toppra only produces clamped cubic splines
    path = ta.SplineInterpolator(x, waypts.copy(), bc_type="clamped")
    # avoid going over limit taking into account toppra's precision
    pc_vel = constraint.JointVelocityConstraint(
        vlim - np.sign(vlim) * V_LIM_EPS
    )

    def _compute_cspl_with_varying_alim(alim_coefficients):
        # Can be either Collocation (0) or Interpolation (1).
        # Interpolation gives more accurate results with
        # slightly higher computational cost
        pc_acc = constraint.JointAccelerationConstraint(
            alim_coefficients.reshape(-1, 1) * alim,
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
        # gridpoints = interpolator.propose_gridpoints(
        #     path, max_err_threshold=max_grid_err
        # )
        instance = algo.TOPPRA(
            [pc_vel, pc_acc],
            path,
            # gridpoints=gridpoints,
            solver_wrapper="seidel",
        )
        jnt_traj = instance.compute_trajectory(0, 0)
        if jnt_traj is None:  # toppra has failed
            if min_pair_dist < DIST_EPS:  # duplicates are probably why
                logger.error(
                    "Duplicates not allowed in input waypoints. "
                    "At least one pair of adjacent waypoints have "
                    f"distance less than epsilon = {DIST_EPS}."
                )
            logger.error(
                f"Failed waypts:\n{waypts}\nvlim:\n{vlim}\nalim:\n{alim}"
            )
            raise RuntimeError("Toppra failed to compute trajectory.")
        cs = jnt_traj.cspl
        # If the initial estimated path length (run time) of the trajectory
        # isn't very close to the actual computed one,
        # (say off by a factor of 2),
        # toppra goes a bit crazy sometimes extends the spline to x >~ 1e3.
        # This cannot be fixed by tweaking the fit.
        # N_samples = 39 may have this behaviour unless x_max > 1.6,
        # while N_samples = 46 may require x_max < 1.6 to be controllable.
        # So we keep x_max towards the small side to guarantee controllability.
        # We check that and truncate the spline at a predefined upper bound.
        if path_length_limit and cs.x[-1] > path_length_limit:
            mask = cs.x <= path_length_limit
            logger.warning(
                "Suboptimal trajectory derived, input waypoints likely "
                "ill-formed. "
                f"x > {path_length_limit:.3f} for "
                f"{(~mask).sum()}/{cs.x.size} ending knots, "
                f"x_max: {cs.x[-1]:.2f}."
            )
            # now truncate and check that it is still close to the original end
            cs = CubicSpline(cs.x[mask], cs(cs.x[mask]), bc_type="natural")
            new_end_pos = cs(cs.x[-1])
            assert np.linalg.norm(new_end_pos - waypts[-1]) < DIST_EPS, (
                f"Truncated CubicSpline, ending at\n{new_end_pos},\n"
                f"no longer arrives at the original ending waypoint\n"
                f"{waypts[-1]}\n"
                f"given DIST_EPS = {DIST_EPS}. "
                "Try closer and smoother waypoints with a smaller path length."
            )
            logger.info(
                f"Optimised CubicSpline truncated at limit x = "
                f"{path_length_limit:.3f}, still arriving at the original "
                f"ending waypoint up to DIST_EPS: {DIST_EPS}. "
                f"Duration after truncation: {cs.x[-1]:.3f}."
            )
        # Toppra goes a bit wider than a precise natural cubic spline.
        # We could find the leftmost and rightmost common roots of all dof,
        # which are the original end points, but that algorithm not guaranteed
        # to converge below a sufficiently small error and is not efficient.
        # We could also use scipy to respline naturally, but it only support
        # one boundary condition, either zero velocity or zero accleration.
        # Manually treat two ends to force both derivatives to be zero.
        impose_natural_bc(cs)
        return cs

    alim_coefficients = np.ones(alim.shape[0])  # no reduction on the first try
    cs = _compute_cspl_with_varying_alim(alim_coefficients)
    if verify_lims:  # flag for checking if vlim is obeyed
        logger.info("Verifying that given limits are strictly obeyed...")
        passed = False  # modify this after each iteration of lim check
        while not passed:
            new_coefficients = _verify_lims(cs, vlim, alim)
            if new_coefficients is None:  # all lims satisfied when None
                passed = True
            else:
                alim_coefficients *= new_coefficients
                logger.info(
                    "Current iteration violates acceleration limits, "
                    "trying new coefficients: "
                    f"{repr(np.around(alim_coefficients, 3))}..."
                )
                cs = _compute_cspl_with_varying_alim(alim_coefficients)

    n_knots = cs.x.size
    logger.info(
        f"Finished computing time-optimised trajectory of {n_knots} knots, "
        f"duration: {cs.x[-1]:.3f} s. "
    )
    logger.warning(
        "To preserve constraints, continuity, and boundary conditions, this "
        "computed CubicSpline polynomial MUST NOT be resplined arbitrarily."
    )
    if return_cs:
        return cs
    return (
        n_knots,
        np.ascontiguousarray(cs.x, dtype=np.float64),
        np.ascontiguousarray(cs.c, dtype=np.float64),
    )


def _find_waypts_indices(waypts, cs):
    """Find the indices for the original waypoints in cubic spline knots."""
    idx = np.zeros(waypts.shape[0], dtype=int)
    k = 0  # index for knots, scan all knots left to right, start at the 0th
    for i, waypt in enumerate(waypts):
        waypt_min_err = float("inf")  # always reset error for current waypt
        while k < cs.x.size:
            err = np.linalg.norm(cs(cs.x[k]) - waypt)
            if err <= waypt_min_err:
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


def run_toppra_jnt_crt(
    waypts_jnt,  # (N, Ndof)
    vlim_jnt,  # (Ndof, 2)
    alim_jnt,  # (Ndof, 2)
    waypts_crt,  # (N, 3)
    vlim_crt,  # (3, 2)
    alim_crt,  # (3, 2)
    return_cs=False,
):
    """Optimise joint-space trajectory with additional cartesian limits."""
    logger.info("Optimising joint-space trajectory...")
    cs_jnt = run_topp(
        waypts_jnt, vlim_jnt, alim_jnt, verify_lims=True, return_cs=return_cs
    )
    logger.info("Optimising Cartesian trajectory...")
    cs_crt = run_topp(
        waypts_crt, vlim_crt, alim_crt, verify_lims=True, return_cs=return_cs
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
    n_knots = cs.x.size
    logger.info(
        f"Finished optimising trajectory of {n_knots} knots "
        f"with combined constraints, duration: {cs.x[-1]:.3f} s. "
    )
    logger.warning(
        "To preserve constraints, continuity, and boundary conditions, this "
        "computed CubicSpline polynomial MUST NOT be resplined arbitrarily."
    )
    if return_cs:
        return cs
    return (
        n_knots,
        np.ascontiguousarray(cs.x, dtype=np.float64),
        np.ascontiguousarray(cs.c, dtype=np.float64),
    )
