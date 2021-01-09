"""Functions for enforcing natural boundary conditions on cubic splines."""
import numpy as np

from .three_segment_spline import ThreeSegmentSpline


def insert_at_start_end_of_cspl(cspl_in, xin, cin, xend, cend):
    """Add given knot points into the given cubic spline."""
    assert np.ndim(xin) == 1, "xin is not vector"
    time_step = 1
    time_delta_start = xin[-1] - cspl_in.x[time_step]
    # shift times of remaining elements
    cspl_in.x[time_step + 1 :] += time_delta_start
    # insert new times:
    cspl_in.x = np.concatenate(
        [cspl_in.x[time_step - 1 : time_step], xin, cspl_in.x[time_step + 1 :]]
    )
    cspl_in.c = np.concatenate([cin, cspl_in.c[:, time_step:, :]], axis=1)
    # last element
    time_step = -1
    # insert new times:
    cspl_in.x = np.concatenate(
        [cspl_in.x[:time_step], xend + cspl_in.x[time_step - 1]]
    )
    cspl_in.c = np.concatenate([cspl_in.c[:, :time_step, :], cend], axis=1)
    return cspl_in


def impose_natural_bc(cspl):
    """Take a clamped CubicSpline (0 first derivative) and add natural bc."""
    # for start polynomial
    # Define Inputs
    Ps = cspl(cspl.x[0])
    Vs = np.zeros(cspl.c.shape[2])
    As = np.zeros(cspl.c.shape[2])
    Pe = cspl(cspl.x[1])
    Ve = cspl(cspl.x[1], 1)
    Ae = cspl(cspl.x[1], 2)
    deltaTimeStart = (cspl.x[1] - cspl.x[0]) * 2
    xstart, cstart = ThreeSegmentSpline(Ps, Vs, As, Pe, Ve, Ae, deltaTimeStart)

    # for end polynomial
    Ps = cspl(cspl.x[-2])
    Vs = cspl(cspl.x[-2], 1)
    As = cspl(cspl.x[-2], 2)
    Pe = cspl(cspl.x[-1])
    Ve = np.zeros(cspl.c.shape[2])
    Ae = np.zeros(cspl.c.shape[2])
    deltaTimeEnd = (cspl.x[-1] - cspl.x[-2]) * 2

    xend, cend = ThreeSegmentSpline(Ps, Vs, As, Pe, Ve, Ae, deltaTimeEnd)

    insert_at_start_end_of_cspl(cspl, xstart, cstart, xend, cend)
    return cspl  # still return even if we've modified in-place
