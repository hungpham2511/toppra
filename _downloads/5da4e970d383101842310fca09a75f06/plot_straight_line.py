"""
Retime a straight path
===============================
"""
import toppra as ta
import toppra.constraint as constraint
import toppra.algorithm as algo
import numpy as np
import matplotlib.pyplot as plt
import time

time.sleep(0.1)

################################################################################
way_pts, vel_limits, accel_limits = np.array([[0, 0, 1], [0.2, 0.3, 0]]), np.array([0.1, 0.2, 0.3]), np.r_[1.0,2,3]
path_scalars = np.linspace(0, 1, len(way_pts))
path = ta.SplineInterpolator(path_scalars, way_pts)

ss = np.linspace(0, 1, 100)
qs = path(np.linspace(0, 1, 100))
for i in range(way_pts.shape[1]):
    plt.plot(ss, qs[:, i])
plt.show()

################################################################################
# Create velocity bounds, then velocity constraint object
vlim = np.vstack((-vel_limits, vel_limits)).T
# Create acceleration bounds, then acceleration constraint object
alim = np.vstack((-accel_limits, accel_limits)).T
pc_vel = constraint.JointVelocityConstraint(vlim)
pc_acc = constraint.JointAccelerationConstraint(
    alim, discretization_scheme=constraint.DiscretizationType.Interpolation)

# Setup a parametrization instance. The keyword arguments are
# optional.
instance = algo.TOPPRA([pc_vel, pc_acc], path, solver_wrapper='seidel')
jnt_traj = instance.compute_trajectory(0, 0)


################################################################################
ts_sample = np.linspace(0, jnt_traj.get_duration(), 100)
qs_sample = jnt_traj.eval(ts_sample)  # sampled joint positions
qds_sample = jnt_traj.evald(ts_sample)  # sampled joint velocities
qdds_sample = jnt_traj.evaldd(ts_sample)  # sampled joint accelerations

for i in range(jnt_traj.dof):
    # plot the i-th joint trajectory
    plt.plot(ts_sample, qds_sample[:, i], c="C{:d}".format(i))
    # plot the i-th joint waypoints
    # plt.plot(data['t_waypts'], way_pts[:, i], 'x', c="C{:d}".format(i))
plt.xlabel("Time (s)")
plt.ylabel("Joint velocity (rad/s^2)")
plt.show()

