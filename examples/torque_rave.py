import toppra
import numpy as np
import matplotlib.pyplot as plt
import openravepy as orpy
toppra.setup_logging("INFO")

# Parameters
N = 300
N_samples = 5
SEED = 9
dof = 11

# Generate random waypoints in R7, then interpolate with a spline to
# obtain a random geometric path.
env = orpy.Environment()
env.Load('data/lab1.env.xml')
robot = env.GetRobots()[0]
robot.SetActiveDOFs(robot.GetActiveManipulator().GetArmIndices())

# Create velocity bounds, then velocity constraint object
vlim_ = robot.GetActiveDOFMaxVel()
vlim = np.vstack((-vlim_, vlim_)).T

np.random.seed(SEED)
jlower, jhigher = robot.GetActiveDOFLimits()
way_pts = ((np.random.rand(N_samples, robot.GetActiveDOF()) - 0.5) * (jhigher - jlower) * 0.8 +
           (jlower + jhigher) / 2)
path = toppra.SplineInterpolator(np.linspace(0, 1, 5), way_pts)
ss = np.linspace(0, 1, N + 1)

pc_vel = toppra.constraint.JointVelocityConstraint(vlim)
pc_torque = toppra.create_rave_torque_path_constraint(
    robot, discretization_scheme=toppra.constraint.DiscretizationType.Interpolation)

instance = toppra.algorithm.TOPPRA([pc_vel, pc_torque], path)
traj, _, sdd = instance.compute_trajectory(0, 0, return_profile=True)
print "Parametrized traj duration {:.5f} sec".format(traj.get_duration())

ts = np.arange(0, traj.get_duration(), 1e-2)
q_vec = traj.eval(ts)
qd_vec = traj.evald(ts)
qdd_vec = traj.evaldd(ts)

fig, axs = plt.subplots(3, 1, sharex=True)
axs[0].plot(ts, q_vec)
axs[0].set_title("Joint position")
axs[1].plot(ts, qd_vec)
axs[1].set_title("Joint velocity")
axs[2].plot(ts, qdd_vec)
axs[2].set_title("Joint acceleration")
plt.tight_layout()
plt.show()

# The below code is only used to analyze the intance. It is not
# necessary to compute thhe time-optimal parameterization.
X = instance.compute_feasible_sets()
K = instance.compute_controllable_sets(0, 0)
_, xs, _ = instance.compute_parameterization(0, 0)
xs = xs ** 2
plt.plot(X[:, 1], "-.", c="green", label="Feasible sets")
plt.plot(X[:, 0], "-.", c="green")
plt.plot(K[:, 0], "--", c="red")
plt.plot(K[:, 1], "--", c="red", label="Controllable sets")
plt.plot(xs, c="blue", label="Velocity profile")
plt.legend()
plt.show()

import IPython
if IPython.get_ipython() is None:
    IPython.embed()
