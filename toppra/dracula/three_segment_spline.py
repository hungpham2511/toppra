from numpy.testing import assert_almost_equal
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PPoly

#### Calculations for stretching one segment:
# Ps - Start Position
# Vs - Start Velocity
# As - Start Acceleration
# Pe - End Position
# Ve - End Velocity
# Ae - End Acceleration
# fin - final time of threesegment spline, if not defined, Jmax is used
def ThreeSegmentSpline(
    Ps, Vs, As, Pe, Ve, Ae, fin=None, Jmax=0.5, debugging=False
):
    # Fixed Parameters:
    if fin is None:
        SafetyTime = 1.0
        # difference in Acceleration between start and end:
        deltaA = np.array(abs(Ae - As))
        deltaA = deltaA.max()
        # calculate final time f:
        f = np.array(deltaA / Jmax * SafetyTime)
    else:
        # set final time f from function input
        f = np.array(fin)

    m = f / 3
    n = f / 3
    o = f / 3
    if debugging:
        print("f =", f)
        print("m =", m)
        print("n =", n)
        print("o =", o)

    # Calculate Coefficients:
    P0 = np.array(Ps)
    V0 = np.array(Vs)
    A0 = np.array(As)
    P3 = np.array(Pe)
    V3 = np.array(Ve)
    A3 = np.array(Ae)
    J0 = (
        A3 * o * (n + o)
        - A0 * (3 * m ** 2 + n * (n + o) + 2 * m * (2 * n + o))
        - 2
        * (
            3 * P0
            - 3 * P3
            + 3 * m * V0
            + 2 * n * V0
            + o * V0
            + n * V3
            + 2 * o * V3
        )
    ) / (m * (m + n) * (m + n + o))
    P1 = (A0 * m ** 2) / 2 + (J0 * m ** 3) / 6 + P0 + m * V0
    V1 = A0 * m + (J0 * m ** 2) / 2 + V0
    A1 = A0 + J0 * m
    J1 = -(
        (
            A3 * o ** 2
            + A1 * (3 * n ** 2 + 6 * n * o + 2 * o ** 2)
            + 6 * (P1 - P3 + (n + o) * V1)
        )
        / (n * (n ** 2 + 3 * n * o + 2 * o ** 2))
    )
    P2 = (A1 * n ** 2) / 2 + (J1 * n ** 3) / 6 + P1 + n * V1
    V2 = A1 * n + (J1 * n ** 2) / 2 + V1
    A2 = A1 + J1 * n
    J2 = -((3 * (A2 * o ** 2 + 2 * (P2 - P3 + o * V2))) / o ** 3)

    if debugging:
        V3test = V2 + A2 * o + (1 / 2) * J2 * (o ** 2)
        A3test = A2 + J2 * o
        print("V3test", V3test)
        print("A3test", A3test)
        print("J0", J0, "J1=", J1, "J2", J2)
        print("P1,V1,A1,J1 = ", P1, V1, A1, J1)
        print("P2,V2,A2,J2 = ", P2, V2, A2, J2)

    # Gather results
    polynomial_coeff_m = np.stack((1 / 6 * J0, 1 / 2 * A0, V0, P0))
    polynomial_coeff_n = np.stack((1 / 6 * J1, 1 / 2 * A1, V1, P1))
    polynomial_coeff_o = np.stack((1 / 6 * J2, 1 / 2 * A2, V2, P2))
    x = np.array([m, m + n, m + n + o])
    c = np.stack(
        [polynomial_coeff_m, polynomial_coeff_n, polynomial_coeff_o], axis=1
    )
    if debugging:
        print("x=\n", x)
        print("c=\n", c)
    return x, c


if __name__ == "__main__":
    ### One Dimensional Test
    # Inputs
    debugging = False
    Jmax = np.array(2)
    Ps = np.array(0.9)
    Vs = np.array(0.0)
    As = np.array(0.0)
    Pe = np.array(40.5)
    Ve = np.array(8.0)
    Ae = np.array(20.1)
    x, c = ThreeSegmentSpline(
        Ps, Vs, As, Pe, Ve, Ae, Jmax, debugging=debugging
    )
    print("x =\n", x)
    print("c =\n", c)
    print("c.shape=\n", c.shape)

    # Prepend 0
    xr = np.insert(x, 0, 0)
    if debugging:
        print(xr)

    # Increase Dimensions
    cr = np.expand_dims(c, axis=2)
    if debugging:
        print(cr)
    cspl = PPoly(cr, xr)
    t_sampled = np.linspace(0, xr[-1], 100)
    y = cspl(t_sampled)
    yd = cspl(t_sampled, 1)
    ydd = cspl(t_sampled, 2)

    assert_almost_equal(y[0], Ps)
    assert_almost_equal(yd[0], Vs)
    assert_almost_equal(ydd[0], As)
    assert_almost_equal(y[-1], Pe)
    assert_almost_equal(yd[-1], Ve)
    assert_almost_equal(ydd[-1], Ae)
    if debugging:
        plt.plot(t_sampled, cspl(t_sampled))
        plt.show()

    ### Multi Dimensional Test
    # Inputs before running ThreeSegmentSpline
    Jmax = 2
    Ps = np.array([0.9, 0.1, 0.5, -4])
    Vs = np.array([0.0, 0.0, 0.0, 0.0])
    As = np.array([0.0, 0.0, 0.0, 0.0])
    Pe = np.array([1.9, 2.2, -0.5, 4])
    Ve = np.array([1.0, 2, 3, -2])
    Ae = np.array([0.9, -0.2, 0.5, 3.4])
    x2, c2 = ThreeSegmentSpline(Ps, Vs, As, Pe, Ve, Ae, Jmax)
    print("x2 =\n", x2)
    print("c2 =\n", c2)
    print("c2.shape=\n", c2.shape)
    # Prepend 0 to x2:
    xr2 = np.insert(x2, 0, 0)
    cr2 = c2
    if debugging:
        print(xr)
    cspl2 = PPoly(cr2, xr2)
    t_sampled2 = np.linspace(0, xr2[-1], 100)
    y2 = cspl2(t_sampled2)
    yd2 = cspl2(t_sampled2, 1)
    ydd2 = cspl2(t_sampled2, 2)

    assert_almost_equal(y2[0], Ps)
    assert_almost_equal(yd2[0], Vs)
    assert_almost_equal(ydd2[0], As)
    assert_almost_equal(y2[-1], Pe)
    assert_almost_equal(yd2[-1], Ve)
    assert_almost_equal(ydd2[-1], Ae)

    if debugging:
        plt.plot(t_sampled2, cspl2(t_sampled2))
        plt.show()
        plt.plot(t_sampled2, cspl2(t_sampled2, 1))
        plt.show()
        plt.plot(t_sampled2, cspl2(t_sampled2, 2))
        plt.show()
