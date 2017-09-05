from toppra import (create_velocity_path_constraint,
                    create_acceleration_path_constraint,
                    create_rave_torque_path_constraint,
                    qpOASESPPSolver,
                    compute_trajectory_gridpoints,
                    smooth_singularities,
                    interpolate_constraint)
from toppra import SplineInterpolator
import numpy as np
import matplotlib.pyplot as plt
import time
from openravepy import Environment


import coloredlogs
coloredlogs.install(level='INFO')
np.set_printoptions(3)

# Parameters
N = 300
N_samples = 5
SEED = 9
dof = 11

# Generate random waypoints in R7, then interpolate with a spline to
# obtain a random geometric path.
env=Environment()
env.Load('data/lab1.env.xml')
robot = env.GetRobots()[0]

np.random.seed(SEED)
jlower, jhigher = robot.GetDOFLimits()
way_pts = ((np.random.rand(N_samples, dof) - 0.5) * (jhigher - jlower) * 0.8 +
           (jlower + jhigher) / 2)
path = SplineInterpolator(np.linspace(0, 1, 5), way_pts)
ss = np.linspace(0, 1, N + 1)

# Create velocity bounds, then velocity constraint object
vlim_ = robot.GetDOFVelocityLimits()
vlim = np.vstack((-vlim_, vlim_)).T
pc_vel = create_velocity_path_constraint(path, ss, vlim)

# Create torque bounds
pc_torque = create_rave_torque_path_constraint(path, ss, robot)
cset = [pc_vel, pc_torque]

# The below show first-order interpolation strategq.
t_ = time.time()
cset_intp = [interpolate_constraint(pc) for pc in cset]
t_intp_interpolate = time.time() - t_
pp_intp = qpOASESPPSolver(cset_intp)
t_intp_setup = time.time() - t_
us_intp, xs_intp = pp_intp.solve_topp()
t_intp_solve = time.time() - t_ - t_intp_setup
t_intp, q_intp, qd_intp, qdd_intp = compute_trajectory_gridpoints(
    path, pp_intp.ss, us_intp, xs_intp)
t_intp_total = time.time() - t_


# Plotting
f, axs = plt.subplots(2, 1)

axs[0].plot(t_intp, qd_intp[:, [1, 2]])
axs[0].plot(t_intp, qd_intp, alpha=0.2)
axs[0].plot(t_intp, qdd_intp[:, [1, 2]])
axs[0].plot(t_intp, qdd_intp, alpha=0.2)

axs[1].plot(np.sqrt(pp_intp.K[:, 0]), '--', c='C3')
axs[1].plot(np.sqrt(pp_intp.K[:, 1]), '--', c='C3')
axs[1].plot(np.sqrt(xs_intp))

plt.tight_layout()
plt.show()

import IPython
if IPython.get_ipython() is None:
    IPython.embed()
