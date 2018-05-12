import toppra as ta
import toppra.constraint as constraint
import toppra.algorithm as algo
import numpy as np
import matplotlib.pyplot as plt
import time
import coloredlogs
coloredlogs.install(level='DEBUG')

# Parameters
N = 300
N_samples = 5
SEED = 9
dof = 7

# Random waypoints used to obtain a random geometric path.
np.random.seed(SEED)
way_pts = np.random.randn(N_samples, dof)

# Create velocity bounds, then velocity constraint object
vlim_ = np.random.rand(dof) * 20
vlim = np.vstack((-vlim_, vlim_)).T
# Create acceleration bounds, then acceleration constraint object
alim_ = np.random.rand(dof) * 2
alim = np.vstack((-alim_, alim_)).T

path = ta.SplineInterpolator(np.linspace(0, 1, 5), way_pts)
pc_vel = constraint.JointVelocityConstraint(vlim)
pc_acc = constraint.JointAccelerationConstraint(
    alim, discretization_scheme=constraint.DiscretizationType.Interpolation)
instance = algo.TOPPRA([pc_vel, pc_acc], path,gridpoints=np.linspace(0, 1, 101), solver_wrapper='qpOASES')

X = instance.compute_feasible_sets()
K = instance.compute_controllable_sets(0, 0)

_, sd_vec, _ = instance.compute_parameterization(0, 0)

X = np.sqrt(X)
K = np.sqrt(K)

plt.plot(X[:, 0], c='green')
plt.plot(X[:, 1], c='green')
plt.plot(K[:, 0], '--', c='red')
plt.plot(K[:, 1], '--', c='red')
plt.plot(sd_vec)
plt.show()

jnt_traj, aux_traj = instance.compute_trajectory(0, 0)
ts_sample = np.linspace(0, jnt_traj.get_duration(), 100)
qs_sample = jnt_traj.evaldd(ts_sample)

plt.plot(ts_sample, qs_sample)
plt.show()
