import numpy as np
import matplotlib.pyplot as plt
from toppra.dracula import RunTopp

# Random Traj
SEED = 1
N_samples = 10
dof = 6
np.random.seed(SEED)
way_pts = np.random.rand(N_samples, dof)
vlim = np.asarray([2]*dof)
vlim = np.transpose(np.vstack((-vlim, vlim)))
alim = np.asarray([2]*dof)
alim = np.transpose(np.vstack((-alim, alim)))
jmax = 30

robot_command_rate = 20
print("Running once")

[topp_breaks_count_final, x,c] = RunTopp(way_pts, vlim, alim, robot_command_rate,return_spline_parameters = True,topp_breaks_count = 1001, debug_active = False)
print("topp_breaks_count_final",topp_breaks_count_final)

print("Running it again")
[x,c,cspl,csplcp] = RunTopp(way_pts, vlim, alim, robot_command_rate,return_spline_parameters = True,topp_breaks_count = 1001, debug_active = True)

# Plotting
s_sampled = np.linspace(0,csplcp.x[-1], 100)
fig, axs = plt.subplots(1, 4, sharex=True, figsize=[18, 4])
for i in range(csplcp.c.shape[2]):
    axs[0].plot(s_sampled, csplcp(s_sampled)[:, i], label="J{:d}".format(i + 1))
    axs[1].plot(s_sampled, csplcp(s_sampled,1)[:, i], label="J{:d}".format(i + 1))
    axs[2].plot(s_sampled, csplcp(s_sampled,2)[:, i], label="J{:d}".format(i + 1))
    axs[3].plot(s_sampled, csplcp(s_sampled,3)[:, i], label="J{:d}".format(i + 1))
axs[0].set_xlabel("Time (s)")
axs[0].set_ylabel("Joint position (rad)")
#     axs[0].legend()
#     axs[1].legend()
axs[1].set_xlabel("Time (s)")
axs[1].set_ylabel("Joint velocity (rad/s)")
axs[2].set_xlabel("Time (s)")
axs[2].set_ylabel("Joint acceleration (rad/s^2)")
axs[3].set_xlabel("Time (s)")
axs[3].set_ylabel("Joint jerk (rad/s^3)")
axs[0].legend()
axs[1].legend()
axs[2].legend()
axs[3].legend()
plt.tight_layout()
fig.suptitle('original')
plt.show()

s_sampled2 = np.linspace(0,cspl.x[-1], 100)
fig, axs = plt.subplots(1, 4, sharex=True, figsize=[18, 4])
for i in range(cspl.c.shape[2]):
    axs[0].plot(s_sampled2, cspl(s_sampled2)[:, i], label="J{:d}".format(i + 1))
    axs[1].plot(s_sampled2, cspl(s_sampled2,1)[:, i], label="J{:d}".format(i + 1))
    axs[2].plot(s_sampled2, cspl(s_sampled2,2)[:, i], label="J{:d}".format(i + 1))
    axs[3].plot(s_sampled2, cspl(s_sampled2,3)[:, i], label="J{:d}".format(i + 1))
axs[0].set_xlabel("Time (s)")
axs[0].set_ylabel("Joint position (rad)")
axs[1].set_xlabel("Time (s)")
axs[1].set_ylabel("Joint velocity (rad/s)")
axs[2].set_xlabel("Time (s)")
axs[2].set_ylabel("Joint acceleration (rad/s^2)")
axs[3].set_xlabel("Time (s)")
axs[3].set_ylabel("Joint jerk (rad/s^3)")
axs[0].legend()
axs[1].legend()
axs[2].legend()
axs[3].legend()
plt.tight_layout()
fig.suptitle('new')
plt.show()