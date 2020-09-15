from .three_segment_spline import ThreeSegmentSpline
from scipy.interpolate import CubicSpline
import numpy as np
import matplotlib.pyplot as plt


def InsertAtStartEndOfSpline(cspl_in, xin, cin, xend, cend, debugging=False):

    assert np.ndim(xin) == 1, "xin is not vector"
    time_step = 1
    time_delta_start = xin[-1] - cspl_in.x[time_step]
    if debugging:
        print("x before replace=", cspl_in.x)
    # shift times of remaining elements
    cspl_in.x[time_step + 1 :] += time_delta_start
    # insert new times:
    if debugging:
        print(
            "\n{}\n{}\n{}".format(
                cspl_in.x[time_step - 1 : time_step],
                xin,
                cspl_in.x[time_step + 1 :],
            )
        )
    cspl_in.x = np.concatenate(
        [cspl_in.x[time_step - 1 : time_step], xin, cspl_in.x[time_step + 1 :]]
    )
    if debugging:
        print("x after replace=", cspl_in.x)

    if debugging:
        print("c[:,:,0] before replace=\n", cspl_in.c[:, :, 0])
        print(cin.shape, "\n", cspl_in.c[:, time_step:, :].shape)
    cspl_in.c = np.concatenate([cin, cspl_in.c[:, time_step:, :]], axis=1)
    if debugging:
        print("c[:,:,0] after replace=\n", cspl_in.c[:, :, 0])

    # last element
    time_step = -1
    # insert new times:
    if debugging:
        print(
            "\n{}\n{}\n{}".format(
                cspl_in.x[time_step - 1 : time_step],
                xin,
                cspl_in.x[time_step + 1 :],
            )
        )
    cspl_in.x = np.concatenate(
        [cspl_in.x[:time_step], xend + cspl_in.x[time_step - 1]]
    )
    if debugging:
        print("x after replace at end=", cspl_in.x)

    # insert new coefficients:
    if debugging:
        print("c[:,:,0] before replace=\n", cspl_in.c[:, :, 0])
        print(cend.shape, "\n", cspl_in.c[:, :time_step, :].shape)
    cspl_in.c = np.concatenate([cspl_in.c[:, :time_step, :], cend], axis=1)
    if debugging:
        print("c[:,:,0] after replace=\n", cspl_in.c[:, :, 0])
    return cspl_in


def ZeroAccelerationAtStartAndEnd(cspl, debugging=False):
    # for start polynomial
    # Define Inputs
    Ps = cspl(cspl.x[0])
    Vs = np.zeros(cspl.c.shape[2])
    As = np.zeros(cspl.c.shape[2])
    Pe = cspl(cspl.x[1])
    Ve = cspl(cspl.x[1], 1)
    Ae = cspl(cspl.x[1], 2)
    if debugging:
        print("Ps, Vs, As = ", Ps, Vs, As)
        print("Pe, Ve, Ae, Jmax = ", Pe, Ve, Ae, Jmax)
    deltaTimeStart = (cspl.x[1] - cspl.x[0]) * 2
    xstart, cstart = ThreeSegmentSpline(
        Ps, Vs, As, Pe, Ve, Ae, deltaTimeStart, debugging=debugging
    )

    if debugging:
        print("x =\n", x)
        print("c.shape =\n", c.shape)
        print("c =\n", c)
    # for end polynomial
    Ps = cspl(cspl.x[-2])
    Vs = cspl(cspl.x[-2], 1)
    As = cspl(cspl.x[-2], 2)
    Pe = cspl(cspl.x[-1])
    Ve = np.zeros(cspl.c.shape[2])
    Ae = np.zeros(cspl.c.shape[2])
    deltaTimeEnd = (cspl.x[-1] - cspl.x[-2]) * 2

    if debugging:
        print("Ps, Vs, As = ", Ps, Vs, As)
        print("Pe, Ve, Ae, Jmax = ", Pe, Ve, Ae, Jmax)
    xend, cend = ThreeSegmentSpline(
        Ps, Vs, As, Pe, Ve, Ae, deltaTimeEnd, debugging=debugging
    )

    if debugging:
        csplcp = copy.deepcopy(cspl)

    InsertAtStartEndOfSpline(cspl, xstart, cstart, xend, cend)
    if debugging:
        plt.figure()
        s_sampled = np.linspace(0, csplcp.x[-1], 100)
        fig, axs = plt.subplots(1, 4, sharex=True, figsize=[18, 4])
        for i in range(csplcp.c.shape[2]):
            axs[0].plot(
                s_sampled, csplcp(s_sampled)[:, i], label="J{:d}".format(i + 1)
            )
            axs[1].plot(
                s_sampled,
                csplcp(s_sampled, 1)[:, i],
                label="J{:d}".format(i + 1),
            )
            axs[2].plot(
                s_sampled,
                csplcp(s_sampled, 2)[:, i],
                label="J{:d}".format(i + 1),
            )
            axs[3].plot(
                s_sampled,
                csplcp(s_sampled, 3)[:, i],
                label="J{:d}".format(i + 1),
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
        plt.show()
        s_sampled2 = np.linspace(0, cspl.x[-1], 100)
        fig, axs = plt.subplots(1, 4, sharex=True, figsize=[18, 4])
        for i in range(cspl.c.shape[2]):
            axs[0].plot(
                s_sampled2, cspl(s_sampled2)[:, i], label="J{:d}".format(i + 1)
            )
            axs[1].plot(
                s_sampled2,
                cspl(s_sampled2, 1)[:, i],
                label="J{:d}".format(i + 1),
            )
            axs[2].plot(
                s_sampled2,
                cspl(s_sampled2, 2)[:, i],
                label="J{:d}".format(i + 1),
            )
            axs[3].plot(
                s_sampled2,
                cspl(s_sampled2, 3)[:, i],
                label="J{:d}".format(i + 1),
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
        plt.show()


if __name__ == "__main__":
    s_array = [0, 3, 8, 12]
    wp_array = [(0, 0, 3, -2), (1, 2, 0, -3), (2, 0, 4, 2), (3, -1, 2, 0)]
    cspl = CubicSpline(s_array, wp_array, bc_type="clamped")
    ZeroAccelerationAtStartAndEnd(cspl)
