"""Wrapper for toppra that takes in waypoints, v and a limits."""
import sys
import warnings

import numpy as np
from scipy.interpolate import CubicSpline

import toppra as ta
import toppra.algorithm as algo
from toppra import constraint  # , interpolator  # needed by custom gridpoints

from .zero_acceleration_start_end import ZeroAccelerationAtStartAndEnd

ta.setup_logging("WARNING")
JOINT_DIST_EPS = 5e-3  # min epsilon for treating two angles the same


def RunTopp(
    waypts,  # ndarray, (N, dof)
    vlim,  # ndarray, (dof, 2)
    alim,  # ndarray, (dof 2)
    # max_grid_err=1e-4,
    return_cspl=False,
    path_length_limit=150,  # 150 is max we've seen, None to disable truncation
):
    """Call toppra obeying velocity and acceleration limits and naturalness.

    No consecutive duplicates in the waypoints allowed.
    Return natural cubic spline coefficients by default.
    """
    # check for duplicates, assert adjacent pairs have distance not equal to 0
    min_pair_dist = np.linalg.norm(np.diff(waypts, axis=0), axis=1).min()
    if min_pair_dist < JOINT_DIST_EPS:  # issue a warning and try anyway
        warnings.warn(
            "Duplicates found in input waypoints. This is not allowed, "
            "especially for the beginning and the end of the trajectory. "
            "Expect Toppra to throw a controllability exception. "
            "Attempting to optimise trajectory anyway...",
            UserWarning,
        )
    N_samples = waypts.shape[0]
    # initial x for toppra's path, essentially normalised time on x axis
    x_max = 2.5 - 2.35 * np.exp(-0.015 * N_samples)  # empirical fit
    x = np.linspace(0, x_max, N_samples)
    # specifying natural here doensn't make a difference
    # toppra only produces clamped cubic splines
    path = ta.SplineInterpolator(x, waypts.copy(), bc_type="clamped")
    pc_vel = constraint.JointVelocityConstraint(vlim)
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
            print(
                "Duplicates not allowed in input waypoints. "
                "At least one pair of adjacent waypoints have distance "
                f"less than epsilon = {JOINT_DIST_EPS} in joint space.",
                file=sys.stderr,
            )
        print(
            f"Failed waypts:\n{waypts}\n" f"vlim:\n{vlim}\n" f"alim:\n{alim}",
            file=sys.stderr,
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
        warnings.warn(
            f"Toppra derived x > {path_length_limit} covering "
            f"{(~mask).sum()}/{cspl.x.size} ending knots, "
            f"x_max: {cspl.x[-1]:.2f}. Input waypoints are likely ill-formed, "
            "resulting in a lengthy trajectory.",
            UserWarning,
        )
        # now truncate and check that it is still close to the original end
        cspl = CubicSpline(cspl.x[mask], cspl(cspl.x[mask]), bc_type="natural")
        new_end_pos = cspl(cspl.x[-1])
        assert np.linalg.norm(new_end_pos - waypts[-1]) < JOINT_DIST_EPS, (
            f"Truncated CubicSpline, ending at\n{new_end_pos},\n"
            f"no longer arrives at the original ending waypoint\n"
            f"{waypts[-1]}\n"
            f"given max allowed single-joint distance {JOINT_DIST_EPS}. "
            "Try another set of closer waypoints with a smaller path length."
        )
        warnings.warn(
            f"Resulitng CubicSpline was truncated at x upperbound: "
            f"{path_length_limit}, but it still arrives at the original "
            f"ending waypoint to max single-joint distance {JOINT_DIST_EPS}."
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
