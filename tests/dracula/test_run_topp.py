"""Unit test for the toppra.dracula.run_topp() wrapper."""

import glob
import os
import unittest

import numpy as np

import toppra.dracula as tdrac


class TestRunTopp(unittest.TestCase):
    """Test RunTopp()."""

    # setup test data only once as they aren't modified
    glob_regex = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_data",
        "test_waypts_jnt_*.txt",
    )
    paths = sorted(glob.glob(glob_regex))
    waypts_list = [np.loadtxt(path) for path in paths]

    @staticmethod
    def _gen_limits(waypts):
        """Generate maximum vlim and alim arrays given waypts.

        Assume the last 7-dof belong to a cobot. The rest is filled with 3.
        """
        n_dof = waypts.shape[1]
        v_max = 3 * np.ones((n_dof, 2))  # init with 3
        a_max = 3 * np.ones((n_dof, 2))  # init with 3
        v_max[-7:, 0] = -tdrac.V_MAX  # fill 7-dof limits
        v_max[-7:, 1] = tdrac.V_MAX
        a_max[-7:, 0] = -tdrac.A_MAX
        a_max[-7:, 1] = tdrac.A_MAX
        v_max[:-7, 0] *= -1
        a_max[:-7, 0] *= -1
        return v_max, a_max

    def test_run_topp_spline_static_data(self):
        """Test run_topp_spline() using static test data."""
        print("Starting test_run_topp_spline_static_data")
        for coeff in [1, 0.5, 0.12]:
            print(f"Testing with limit reduction coefficient: {coeff}...")
            for i, waypts in enumerate(self.waypts_list):
                print(f"Testing waypoints file {i}...")
                v_max, a_max = TestRunTopp._gen_limits(waypts)
                tdrac.run_topp_spline(
                    waypts,
                    coeff * v_max,
                    coeff * a_max,
                    verify_lims=True,
                )

    def test_run_topp_const_accel_static_data(self):
        """Test run_topp_const_accel() using static test data."""
        print("Starting test_run_topp_const_accel_static_data")
        for coeff in [1, 0.5, 0.12]:
            print(f"Testing with limit reduction coefficient: {coeff}...")
            for i, waypts in enumerate(self.waypts_list):
                print(f"Testing waypoints file {i}...")
                v_max, a_max = TestRunTopp._gen_limits(waypts)
                tdrac.run_topp_const_accel(
                    waypts,
                    coeff * v_max,
                    coeff * a_max,
                    cmd_rate=1000,
                    verify_lims=True,
                )

    @staticmethod
    def test_run_topp_spline_random_data():
        """Test run_topp_spline() using randoms."""
        # 2000 waypts supported but can be commented out for speed if needed
        n_dof = 7
        vlim = np.asarray([1] * n_dof)
        alim = np.asarray([2] * n_dof)
        vlim = np.vstack([-vlim, vlim]).T
        alim = np.vstack([-alim, alim]).T
        for n_waypts in [2, 20, 50, 200]:  # , 2000]:
            print(f"Testing {n_waypts} random waypoints...")
            waypts = np.random.rand(n_waypts, n_dof)
            tdrac.run_topp_spline(waypts, vlim, alim, verify_lims=True)


if __name__ == "__main__":
    unittest.main()

    # import matplotlib.pyplot as plt
    # # test using static test data
    # v_max = np.vstack([-V_MAX, V_MAX]).T
    # a_max = np.vstack([-A_MAX, A_MAX]).T
    # # two sets of vlims, alims, two reduction coefficients (safety factor)
    # for coeff in [1, 0.5, 0.2, 0.1, 0.05]:
    #     print(f"limit reduction coefficient: {coeff}")
    #     for i in range(5):
    #         print(f"testing waypoints file {i}...")
    #         waypts = np.loadtxt(
    #             f"/src/toppra/tests/dracula/test_waypts_jnt_{i}.txt"
    #         )  # (33, 7)
    #         _ = run_topp(
    #             waypts, coeff * v_max, coeff * a_max, verify_lims=True
    #         )  # assert no throw

    # # test using randoms
    # # 2000 is supported but commented out for speed
    # for n in [2, 20, 50, 200]:  # , 2000]:
    #     print(f"Testing {n} random waypoints with no truncation...")
    #     topp_breaks_count_final, _, _ = run_topp_random(n, False)

    # # Plotting
    # csplcp = copy.deepcopy(cspl)
    # s_sampled = np.linspace(0, csplcp.x[-1], 100)
    # fig, axs = plt.subplots(1, 4, sharex=True, figsize=[18, 4])
    # for i in range(csplcp.c.shape[2]):
    #     axs[0].plot(
    #         s_sampled, csplcp(s_sampled)[:, i],
    #         label="J{:d}".format(i + 1)
    #     )
    #     axs[1].plot(
    #         s_sampled, csplcp(s_sampled, 1)[:, i],
    #         label="J{:d}".format(i + 1)
    #     )
    #     axs[2].plot(
    #         s_sampled, csplcp(s_sampled, 2)[:, i],
    #         label="J{:d}".format(i + 1)
    #     )
    #     axs[3].plot(
    #         s_sampled, csplcp(s_sampled, 3)[:, i],
    #         label="J{:d}".format(i + 1)
    #     )
    # axs[0].set_xlabel("Time (s)")
    # axs[0].set_ylabel("Joint position (rad)")
    # axs[1].set_xlabel("Time (s)")
    # axs[1].set_ylabel("Joint velocity (rad/s)")
    # axs[2].set_xlabel("Time (s)")
    # axs[2].set_ylabel("Joint acceleration (rad/s^2)")
    # axs[3].set_xlabel("Time (s)")
    # axs[3].set_ylabel("Joint jerk (rad/s^3)")
    # axs[0].legend()
    # axs[1].legend()
    # axs[2].legend()
    # axs[3].legend()
    # plt.tight_layout()
    # fig.suptitle("original")
    # # plt.show()

    # s_sampled2 = np.linspace(0, cspl.x[-1], 100)
    # fig, axs = plt.subplots(1, 4, sharex=True, figsize=[18, 4])
    # for i in range(cspl.c.shape[2]):
    #     axs[0].plot(
    #         s_sampled2, cspl(s_sampled2)[:, i],
    #         label="J{:d}".format(i + 1)
    #     )
    #     axs[1].plot(
    #         s_sampled2, cspl(s_sampled2, 1)[:, i],
    #         label="J{:d}".format(i + 1)
    #     )
    #     axs[2].plot(
    #         s_sampled2, cspl(s_sampled2, 2)[:, i],
    #         label="J{:d}".format(i + 1)
    #     )
    #     axs[3].plot(
    #         s_sampled2, cspl(s_sampled2, 3)[:, i],
    #         label="J{:d}".format(i + 1)
    #     )
    # axs[0].set_xlabel("Time (s)")
    # axs[0].set_ylabel("Joint position (rad)")
    # axs[1].set_xlabel("Time (s)")
    # axs[1].set_ylabel("Joint velocity (rad/s)")
    # axs[2].set_xlabel("Time (s)")
    # axs[2].set_ylabel("Joint acceleration (rad/s^2)")
    # axs[3].set_xlabel("Time (s)")
    # axs[3].set_ylabel("Joint jerk (rad/s^3)")
    # axs[0].legend()
    # axs[1].legend()
    # axs[2].legend()
    # axs[3].legend()
    # plt.tight_layout()
    # fig.suptitle("new")
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
    #                 s_sampled,
    #                 csplcp(s_sampled)[:, i], label="J{:d}".format(i + 1)
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
    #                 s_sampled2,
    #                 cspl(s_sampled2)[:, i], label="J{:d}".format(i + 1)
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
