from .interpolator import RaveTrajectoryWrapper, SplineInterpolator
from .constraint import JointAccelerationConstraint, JointVelocityConstraint, DiscretizationType
from .algorithm import TOPPRA
import numpy as np
import logging

logger = logging.getLogger(__name__)


def retime_active_joints_kinematics(traj, robot, output_interpolator=False, vmult=1.0, amult=1.0, N=100, use_ravewrapper=False):
    """ Retime a trajectory wrt velocity and acceleration constraints.

    Parameters
    ----------
    traj: OpenRAVE Trajectory
    robot: OpenRAVE Robot
    output_interpolator: bool, optional
        If this is true, output the interpolator object together with the retimed trajectory.
    vmult: float, optional
        Safety factor.
    amult: float, optional
        Safety factor.
    N: int, optional
        Approximate number of gridpoints.
    use_ravewrapper: bool, optional
        If is true, use the input openrave trajectory directly for parameterization. This is,
        however, not desirable because the input trajectory tends to have discontinous second
        order derivative. This causes alots of problems for the parameterization algorithm
        TOPPRA.

    Returns
    -------
    traj_rave: OpenRAVE Trajectory
        The retimed trajectory. Return None if retiming fails.
    traj_ra: SplineInterpolator
        Return if 'output_interpolator' is True.
    """
    logger.info("Start retiming an OpenRAVE trajectory.")
    if use_ravewrapper:
        logger.warn("Use RaveTrajectoryWrapper. This might not work properly!")
        path = RaveTrajectoryWrapper(traj, robot)
    else:
        logger.info("Use a spline to represent the input path!")
        path = RaveTrajectoryWrapper(traj, robot)
        waypoints = path.eval(path.ss_waypoints)
        ss_waypoints = [0]
        for i in range(waypoints.shape[0] - 1):
            ss_waypoints.append(ss_waypoints[-1] + np.linalg.norm(waypoints[i + 1] - waypoints[i]))
        path = SplineInterpolator(ss_waypoints, waypoints)

    vmax = robot.GetActiveDOFMaxVel() * vmult
    amax = robot.GetActiveDOFMaxAccel() * amult
    vlim = np.vstack((-vmax, vmax)).T
    alim = np.vstack((-amax, amax)).T

    pc_vel = JointVelocityConstraint(vlim)
    pc_acc = JointAccelerationConstraint(
        alim, discretization_scheme=DiscretizationType.Interpolation)

    # Include the waypoints in the grid
    ds = path.ss_waypoints[-1] / N
    gridpoints = [path.ss_waypoints[0]]
    for i in range(len(path.ss_waypoints) - 1):
        Ni = int(np.ceil((path.ss_waypoints[i + 1] - path.ss_waypoints[i]) / ds))
        gridpoints.extend(
            path.ss_waypoints[i]
            + np.linspace(0, 1, Ni + 1)[1:] * (path.ss_waypoints[i + 1] - path.ss_waypoints[i]))
    instance = TOPPRA([pc_vel, pc_acc], path, gridpoints=gridpoints, solver_wrapper='qpOASES')

    traj_ra, aux_traj = instance.compute_trajectory(0, 0)
    if traj_ra is None:
        logger.warn("Retime fails. Something is wrong!")
        traj_rave = None
    else:
        logger.info("Retime successes!")
        traj_rave = traj_ra.compute_rave_trajectory(robot)

    if output_interpolator:
        return traj_rave, traj_ra
    else:
        return traj_rave

