"""
Retime a path subject to kinematic constraints
==============================================
"""

import toppra as ta
import toppra.constraint as constraint
import toppra.algorithm as algo
import numpy as np
import matplotlib.pyplot as plt
import time

ta.setup_logging("INFO")


def generate_new_problem(seed=9):
    # Parameters
    N_samples = 5
    dof = 7
    np.random.seed(seed)
    way_pts = np.random.randn(N_samples, dof)
    return (
        np.linspace(0, 1, 5),
        way_pts,
        10 + np.random.rand(dof) * 20,
        10 + np.random.rand(dof) * 2,
    )


ss, way_pts, vlims, alims = generate_new_problem()
path = ta.SplineInterpolator(ss, way_pts)
pc_vel = constraint.JointVelocityConstraint(vlims)
pc_acc = constraint.JointAccelerationConstraint(alims)
instance = algo.TOPPRA([pc_vel, pc_acc], path, parametrizer="ParametrizeConstAccel")
jnt_traj = instance.compute_trajectory()


################################################################################
ts_sample = np.linspace(0, jnt_traj.duration, 100)
qs_sample = jnt_traj(ts_sample)
for i in range(path.dof):
    # plot the i-th joint trajectory
    plt.plot(ts_sample, qs_sample[:, i], c="C{:d}".format(i))
plt.xlabel("Time (s)")
plt.ylabel("Joint position (rad/s^2)")
plt.show()


################################################################################
instance.compute_feasible_sets()
instance.inspect()
