import copy

import matplotlib.pyplot as plt
import numpy as np
from toppra.dracula import RunTopp


def run_toppra_random(N_samples=30, return_cspl=False):
    """Random Traj."""
    dof = 7
    knots = np.random.rand(N_samples, dof)
    vlim = np.asarray([2] * dof)
    alim = np.asarray([2] * dof)
    vlim = np.vstack([-vlim, vlim]).T
    alim = np.vstack([-alim, alim]).T
    return RunTopp(knots, vlim, alim, return_cspl=return_cspl)


for n in [2, 20, 50, 200, 2000]:
    print(f"Testing {n} random waypoints...")
    topp_breaks_count_final, _, _ = run_toppra_random(n, False)

cspl = run_toppra_random(return_cspl=True)

# Plotting
csplcp = copy.deepcopy(cspl)
s_sampled = np.linspace(0, csplcp.x[-1], 100)
fig, axs = plt.subplots(1, 4, sharex=True, figsize=[18, 4])
for i in range(csplcp.c.shape[2]):
    axs[0].plot(
        s_sampled, csplcp(s_sampled)[:, i], label="J{:d}".format(i + 1)
    )
    axs[1].plot(
        s_sampled, csplcp(s_sampled, 1)[:, i], label="J{:d}".format(i + 1)
    )
    axs[2].plot(
        s_sampled, csplcp(s_sampled, 2)[:, i], label="J{:d}".format(i + 1)
    )
    axs[3].plot(
        s_sampled, csplcp(s_sampled, 3)[:, i], label="J{:d}".format(i + 1)
    )
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
fig.suptitle("original")
# plt.show()

s_sampled2 = np.linspace(0, cspl.x[-1], 100)
fig, axs = plt.subplots(1, 4, sharex=True, figsize=[18, 4])
for i in range(cspl.c.shape[2]):
    axs[0].plot(
        s_sampled2, cspl(s_sampled2)[:, i], label="J{:d}".format(i + 1)
    )
    axs[1].plot(
        s_sampled2, cspl(s_sampled2, 1)[:, i], label="J{:d}".format(i + 1)
    )
    axs[2].plot(
        s_sampled2, cspl(s_sampled2, 2)[:, i], label="J{:d}".format(i + 1)
    )
    axs[3].plot(
        s_sampled2, cspl(s_sampled2, 3)[:, i], label="J{:d}".format(i + 1)
    )
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
fig.suptitle("new")
# plt.show()


# test using two knot files
vlim = np.array([0.9689, 0.9689, 0.9689, 0.9689, 1.1627, 1.1627, 1.1627])
alim = np.array([6.6825, 3.3412, 4.4550, 5.5687, 6.6825, 8.9100, 8.9100])
vlim = np.vstack([-vlim, vlim]).T
alim = np.vstack([-alim, alim]).T
print("testing knots file 1")
knots_1 = np.loadtxt("/src/toppra/tests/dracula/test_knots_1.txt")  # (33, 7)
_ = RunTopp(knots_1, vlim, alim)  # assert no throw
print("testing knots file 2")
knots_2 = np.loadtxt("/src/toppra/tests/dracula/test_knots_2.txt")  # (46, 7)
_ = RunTopp(knots_2, vlim, alim)  # assert no throw

# more debugging plots from code files

# if debug_active:
#     print("yay we made an instance")
#     X = instance.compute_feasible_sets()
#     K = instance.compute_controllable_sets(0, 0)
#     _, sd_vec, _ = instance.compute_parameterization(0, 0)
#     X = np.sqrt(X)
#     K = np.sqrt(K)
#     plt.plot(X[:, 0], c="green", label="Feasible sets")
#     plt.plot(X[:, 1], c="green")
#     plt.plot(K[:, 0], "--", c="red", label="Controllable sets")
#     plt.plot(K[:, 1], "--", c="red")
#     plt.plot(sd_vec, label="Velocity profile")
#     plt.title("Path-position path-velocity plot")
#     plt.xlabel("Path position")
#     plt.ylabel("Path velocity square")
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

#    if debugging:
#         plt.figure()
#         s_sampled = np.linspace(0, csplcp.x[-1], 100)
#         fig, axs = plt.subplots(1, 4, sharex=True, figsize=[18, 4])
#         for i in range(csplcp.c.shape[2]):
#             axs[0].plot(
#                 s_sampled, csplcp(s_sampled)[:, i], label="J{:d}".format(i + 1)
#             )
#             axs[1].plot(
#                 s_sampled,
#                 csplcp(s_sampled, 1)[:, i],
#                 label="J{:d}".format(i + 1),
#             )
#             axs[2].plot(
#                 s_sampled,
#                 csplcp(s_sampled, 2)[:, i],
#                 label="J{:d}".format(i + 1),
#             )
#             axs[3].plot(
#                 s_sampled,
#                 csplcp(s_sampled, 3)[:, i],
#                 label="J{:d}".format(i + 1),
#             )
#         axs[0].set_xlabel("Time (s)")
#         axs[0].set_ylabel("Joint position (rad)")
#         axs[1].set_xlabel("Time (s)")
#         axs[1].set_ylabel("Joint velocity (rad/s)")
#         axs[2].set_xlabel("Time (s)")
#         axs[2].set_ylabel("Joint acceleration (rad/s^2)")
#         axs[3].set_xlabel("Time (s)")
#         axs[3].set_ylabel("Joint jerk (rad/s^3)")
#         axs[0].legend()
#         axs[1].legend()
#         axs[2].legend()
#         axs[3].legend()
#         plt.tight_layout()
#         fig.suptitle("original")
#         plt.show()
#         s_sampled2 = np.linspace(0, cspl.x[-1], 100)
#         fig, axs = plt.subplots(1, 4, sharex=True, figsize=[18, 4])
#         for i in range(cspl.c.shape[2]):
#             axs[0].plot(
#                 s_sampled2, cspl(s_sampled2)[:, i], label="J{:d}".format(i + 1)
#             )
#             axs[1].plot(
#                 s_sampled2,
#                 cspl(s_sampled2, 1)[:, i],
#                 label="J{:d}".format(i + 1),
#             )
#             axs[2].plot(
#                 s_sampled2,
#                 cspl(s_sampled2, 2)[:, i],
#                 label="J{:d}".format(i + 1),
#             )
#             axs[3].plot(
#                 s_sampled2,
#                 cspl(s_sampled2, 3)[:, i],
#                 label="J{:d}".format(i + 1),
#             )
#         axs[0].set_xlabel("Time (s)")
#         axs[0].set_ylabel("Joint position (rad)")
#         axs[1].set_xlabel("Time (s)")
#         axs[1].set_ylabel("Joint velocity (rad/s)")
#         axs[2].set_xlabel("Time (s)")
#         axs[2].set_ylabel("Joint acceleration (rad/s^2)")
#         axs[3].set_xlabel("Time (s)")
#         axs[3].set_ylabel("Joint jerk (rad/s^3)")
#         axs[0].legend()
#         axs[1].legend()
#         axs[2].legend()
#         axs[3].legend()
#         plt.tight_layout()
#         fig.suptitle("new")
#         plt.show()
