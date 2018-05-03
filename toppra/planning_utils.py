from .interpolator import RaveTrajectoryWrapper
from .constraint import JointAccelerationConstraint, JointVelocityConstraint, DiscretizationType
from .algorithm import TOPPRA
import numpy as np
import logging

logger = logging.getLogger(__name__)

def retime_active_joints_kinematics(traj, robot, output_interpolator=False, vmult=1.0, amult=1.0, N=100):
    """

    Parameters
    ----------
    traj
    robot
    output_interpolator

    Returns
    -------

    """
    path = RaveTrajectoryWrapper(traj, robot)
    vmax = robot.GetActiveDOFMaxVel() * vmult
    amax = robot.GetActiveDOFMaxAccel() * amult
    vlim = np.vstack((-vmax, vmax)).T
    alim = np.vstack((-amax, amax)).T

    pc_vel = JointVelocityConstraint(vlim)
    pc_acc = JointAccelerationConstraint(
        alim, discretization_scheme=DiscretizationType.Interpolation)

    instance = TOPPRA([pc_vel, pc_acc], path,
                      gridpoints=np.linspace(0, path.get_duration(), N+1),
                      solver_wrapper='qpOASES')

    traj_ra, aux_traj = instance.compute_trajectory(0, 0)
    if traj_ra is None:
        logger.warn("Parameterization fails. Something is wrong!")
        traj_rave = None
    else:
        traj_rave = traj_ra.compute_rave_trajectory(robot)

    if output_interpolator:
        return traj_rave, traj_ra
    else:
        return traj_rave

