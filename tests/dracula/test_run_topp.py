import copy

import matplotlib.pyplot as plt
import numpy as np

from toppra.dracula import A_MAX, V_MAX, RunTopp


def run_toppra_random(N_samples=30, return_cs=False):
    """Random Traj."""
    dof = 7
    rand_waypts = np.random.rand(N_samples, dof)
    vlim = np.asarray([1] * dof)
    alim = np.asarray([2] * dof)
    vlim = np.vstack([-vlim, vlim]).T
    alim = np.vstack([-alim, alim]).T
    return RunTopp(
        rand_waypts, vlim, alim, return_cs=return_cs, verify_lims=True
    )


if __name__ == "__main__":
    # test using static test data
    v_max = np.vstack([-V_MAX, V_MAX]).T
    a_max = np.vstack([-A_MAX, A_MAX]).T
    # two sets of vlims, alims, two reduction coefficients (safety factor)
    for coeff in [1, 0.5, 0.2, 0.1, 0.05]:
        print(f"limit reduction coefficient: {coeff}")
        for i in range(5):
            print(f"testing waypoints file {i}...")
            waypts = np.loadtxt(
                f"/src/toppra/tests/dracula/test_waypts_{i}.txt"
            )  # (33, 7)
            _ = RunTopp(
                waypts, coeff * v_max, coeff * a_max, verify_lims=True
            )  # assert no throw

    # test using randoms
    for n in [2, 20, 50, 200, 2000]:
        print(f"Testing {n} random waypoints with no truncation...")
        topp_breaks_count_final, _, _ = run_toppra_random(n, False)

    cspl = run_toppra_random(return_cs=True)

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
