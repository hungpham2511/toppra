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

from .zero_acceleration_start_end import ZeroAccelerationAtStartAndEnd

ta.setup_logging("INFO")
JOINT_DIST_EPS = 2e-3  # min epsilon for treating two angles the same
# https://frankaemika.github.io/docs/control_parameters.html#constants
V_MAX = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100])
A_MAX = np.array([15, 7.5, 10, 12.5, 15, 20, 20])
DATA_DIR = "/data/toppra"
os.makedirs(DATA_DIR, exist_ok=True)
logger = logging.getLogger("toppra")


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


def RunTopp(
    waypts,  # ndarray, (N, dof)
    vlim,  # ndarray, (dof, 2)
    alim,  # ndarray, (dof 2)
    # max_grid_err=1e-4,
    return_cspl=False,
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
    if min_pair_dist < JOINT_DIST_EPS:  # issue a warning and try anyway
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
    # 2e-3 is as precise as toppra can be, avoid going over limit
    pc_vel = constraint.JointVelocityConstraint(vlim - JOINT_DIST_EPS)
    # Can be either Collocation (0) or Interpolation (1). Interpolation gives
    # more accurate results with slightly higher computational cost
    pc_acc = constraint.JointAccelerationConstraint(
        alim, discretization_scheme=constraint.DiscretizationType.Interpolation
    )
    # use the default gridpoints=None to let interpolator.propose_gridpoints
    # calculate grid that sufficiently covers the path.
    # this ensures the instance is controllable and avoids error:
    #     "An error occurred when computing controllable velocities.
    #     The path is not controllable, or is badly conditioned.
    #     Error: Instance is not controllable"
    # if using clamped as boundary condition, the default gridpoints error
    # 1e-3 is ok and we don't need to calculate gridpoints
    # boundary condition "natural" especially needs support by smaller error
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
        if min_pair_dist < JOINT_DIST_EPS:  # duplicates are probably why
            logger.error(
                "Duplicates not allowed in input waypoints. "
                "At least one pair of adjacent waypoints have "
                f"joint-space distance less than epsilon = {JOINT_DIST_EPS}."
            )
        logger.error(
            f"Failed waypts:\n{waypts}\n" f"vlim:\n{vlim}\n" f"alim:\n{alim}"
        )
        raise RuntimeError("Toppra failed to compute trajectory.")
    cspl = jnt_traj.cspl
    # If the initial estimated path length (run time) of the trajectory isn't
    # very close to the actual computed one (say off by a factor of 2),
    # toppra goes a bit crazy sometimes extends the spline to x >~ 1e3.
    # This cannot be fixed by tweaking the fit:
    # N_samples = 39 may have this behaviour unless x_max > 1.6,
    # while N_samples = 46 may require x_max < 1.6 to be controllable.
    # So we keep x_max towards the small side to guarantee controllability.
    # We check that and truncate the spline at a predefined upper bound.
    if path_length_limit and cspl.x[-1] > path_length_limit:
        mask = cspl.x <= path_length_limit
        logger.warning(
            f"Toppra derived x > {path_length_limit:.3f} covering "
            f"{(~mask).sum()}/{cspl.x.size} ending knots, "
            f"x_max: {cspl.x[-1]:.2f}. Input waypoints are likely ill-formed, "
            "resulting in a suboptimal trajectory."
        )
        # now truncate and check that it is still close to the original end
        cspl = CubicSpline(cspl.x[mask], cspl(cspl.x[mask]), bc_type="natural")
        new_end_pos = cspl(cspl.x[-1])
        assert np.linalg.norm(new_end_pos - waypts[-1]) < JOINT_DIST_EPS, (
            f"Truncated CubicSpline, ending at\n{new_end_pos},\n"
            f"no longer arrives at the original ending waypoint\n"
            f"{waypts[-1]}\n"
            f"given max allowed joint-space distance {JOINT_DIST_EPS}. "
            "Try another set of closer waypoints with a smaller path length."
        )
        logger.info(
            f"Resulitng CubicSpline was truncated at x upperbound: "
            f"{path_length_limit:.3f}, but it still arrives at the original "
            f"ending waypoint to max joint-space distance {JOINT_DIST_EPS}. "
            f"New duration after truncation: {cspl.x[-1]:.3f}."
        )
    # Toppra goes a bit wider than a precise natural cubic spline
    # we could find the leftmost and rightmost common roots of all dof
    # which are the original end points, but that algorithm not guaranteed
    # to converge below a sufficiently small error and is not efficient.
    # We could also use scipy to respline naturally,
    # this brings accel to down 1e-15, but zero velocity is lost (now 1e-3).
    # cspl = CubicSpline(jnt_traj.cspl.x,
    #                    jnt_traj.cspl(jnt_traj.cspl.x),
    #                    bc_type='natural')
    # Manually treat two ends by moving two points next to ends.
    ZeroAccelerationAtStartAndEnd(cspl)
    if return_cspl:
        return cspl
    return (
        len(cspl.x),
        np.ascontiguousarray(cspl.x, dtype=np.float64),
        np.ascontiguousarray(cspl.c, dtype=np.float64),
    )
