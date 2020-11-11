import numpy as np
import toppra as ta
import toppra.algorithm as algo
from toppra import constraint, interpolator

from .zero_acceleration_start_end import ZeroAccelerationAtStartAndEnd

ta.setup_logging("WARNING")


def length_parameterize(points, vmax):
    dist = points.copy()
    dist[1:] = dist[:-1]
    dist = points - dist
    dist = np.sqrt(dist ** 2)
    t_knots = np.abs(dist / vmax)
    t_knots = np.cumsum(t_knots, axis=0)
    return np.max(t_knots, axis=-1)


def RunTopp(
    waypts,  # ndarray, (N, dof)
    vlim,  # ndarray, (dof, 2)
    alim,  # ndarray, (dof 2)
    # max_grid_err=1e-4,
    return_cspl=False,
):
    N_samples = waypts.shape[0]

    # positive v max
    v_max = vlim[:, 1]

    # scale v_max by 0.25 - arbitrary
    t = length_parameterize(waypts, v_max / 0.25)

    # specifying natural here doensn't make a difference
    # toppra only produces clamped cubic splines
    path = ta.SplineInterpolator(t, waypts.copy(), bc_type="clamped")
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
    if jnt_traj is None:
        print(f"Failed waypts:\n{waypts}\n" f"vlim:\n{vlim}\n" f"alim:\n{alim}")
        raise RuntimeError("Toppra failed to compute trajectory.")
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
    ZeroAccelerationAtStartAndEnd(jnt_traj.cspl)
    if return_cspl:
        return jnt_traj.cspl
    return (
        len(jnt_traj.cspl.x),
        np.ascontiguousarray(jnt_traj.cspl.x, dtype=np.float64),
        np.ascontiguousarray(jnt_traj.cspl.c, dtype=np.float64),
    )
