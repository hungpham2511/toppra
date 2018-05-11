from .interpolator import RaveTrajectoryWrapper, SplineInterpolator
from .constraint import JointAccelerationConstraint, JointVelocityConstraint, DiscretizationType
from .algorithm import TOPPRA
import numpy as np
import logging, time

logger = logging.getLogger(__name__)


def retime_active_joints_kinematics(traj, robot, output_interpolator=False, vmult=1.0, amult=1.0, N=100, use_ravewrapper=False, additional_constraints=[]):
    """ Retime a trajectory wrt velocity and acceleration constraints.

    Parameters
    ----------
    traj: OpenRAVE Trajectory or (N,dof)array
        The original trajectory. If is an array, a cubic spline will be used to interpolate
        through all points before parameterization.
    robot: OpenRAVE Robot
        The kinematic limits, which are velocity and acceleration limits, are taken from the robot
        model.
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
    additional_constraints: list, optional
        List of additional constraints to consider.

    Returns
    -------
    traj_rave: OpenRAVE Trajectory
        The retimed trajectory. Return None if retiming fails.
    traj_ra: SplineInterpolator
        Return if 'output_interpolator' is True.
    """
    _t0 = time.time()
    logger.info("Start retiming an OpenRAVE trajectory.")
    if isinstance(traj, np.ndarray):
        logger.info("Received a list of waypoints.")
        ss_waypoints = np.linspace(0, 1, len(traj))
        path = SplineInterpolator(ss_waypoints, traj)
    elif use_ravewrapper:
        logger.warn("Use RaveTrajectoryWrapper. This might not work properly!")
        path = RaveTrajectoryWrapper(traj, robot)
    else:
        logger.info("Use a spline to represent the input path!")
        ss_waypoints = []
        waypoints = []
        spec = traj.GetConfigurationSpecification()
        for i in range(traj.GetNumWaypoints()):
            data = traj.GetWaypoint(i)
            dt = spec.ExtractDeltaTime(data)
            if dt > 1e-5 or len(waypoints) == 0:  # If delta is too small, skip it.
                if len(ss_waypoints) == 0:
                    ss_waypoints.append(0)
                else:
                    ss_waypoints.append(ss_waypoints[-1] + dt)
                waypoints.append(spec.ExtractJointValues(data, robot, robot.GetActiveDOFIndices()))
        path = SplineInterpolator(ss_waypoints, waypoints)

    vmax = robot.GetActiveDOFMaxVel() * vmult
    amax = robot.GetActiveDOFMaxAccel() * amult
    vlim = np.vstack((-vmax, vmax)).T
    alim = np.vstack((-amax, amax)).T

    pc_vel = JointVelocityConstraint(vlim)
    pc_acc = JointAccelerationConstraint(
        alim, discretization_scheme=DiscretizationType.Interpolation)
    logger.info("Number of constraints {:d}".format(2 + len(additional_constraints)))
    logger.info(str(pc_vel))
    logger.info(str(pc_acc))
    for _c in additional_constraints:
        logger.info(str(_c))

    # Include the waypoints in the grid
    ds = path.ss_waypoints[-1] / N
    gridpoints = [path.ss_waypoints[0]]
    for i in range(len(path.ss_waypoints) - 1):
        Ni = int(np.ceil((path.ss_waypoints[i + 1] - path.ss_waypoints[i]) / ds))
        gridpoints.extend(
            path.ss_waypoints[i]
            + np.linspace(0, 1, Ni + 1)[1:] * (path.ss_waypoints[i + 1] - path.ss_waypoints[i]))
    instance = TOPPRA([pc_vel, pc_acc] + additional_constraints, path, gridpoints=gridpoints, solver_wrapper='qpOASES')
    _t1 = time.time()

    traj_ra, aux_traj = instance.compute_trajectory(0, 0)
    _t2 = time.time()
    logger.info("t_setup={:.5f}ms, t_solve={:.5f}ms, t_total={:.5f}ms".format(
        (_t1 - _t0) * 1e3, (_t2 - _t1)*1e3, (_t2 - _t0)*1e3
    ))
    if traj_ra is None:
        logger.warn("Retime fails.")
        traj_rave = None
    else:
        logger.info("Retime successes!")
        traj_rave = traj_ra.compute_rave_trajectory(robot)

    if output_interpolator:
        return traj_rave, traj_ra
    else:
        return traj_rave

