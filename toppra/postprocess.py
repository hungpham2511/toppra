import numpy as np
import quadprog
from scipy.interpolate import PPoly

# def poly_trajectory()


def poly_velocity_profile(ss, sds, m=50, k=3, eps=1e-6):
    """Compute a piecewise polynomial approximating the velocity profile.

    From the output of :class:`.qpOAsesppsolver`, which are two arrays
    `us` and `sds`, this function output a piecewise polynomial
    approximating the function :math:`\dot s(s)`.

    Various options are available. See belows for more details.

    Parameters
    ----------
    ss : array
        Shaped (N+1, ). Grid points used to solve the TOPP proble.
    sds : array
        Shaped (N+1, ). Velocities at each grid point. Obtain
        from TOPP-RA.
    m : int, optional
        Number of polynomials.
        The polynomials' domains are equally spaced intervals over
        the line :math:`[s_0, s_N]`.
    k : int, optional
        Order of the polynomial.
    eps : float, optional
        Regularization weight.

    Returns
    -------
    out : :class:`scipy.interpolate.PPoly`
        FIXME

    Notes
    -----

    To find the coefficients of the polynomials, we formulate and
    solve a quadratic program. The variables are concatenated
    coefficients of the polynomials. The objective is MSE with `sds`.
    We consider different constraints:

    - end-points conditions;
    - C0 and C1 continuity;
    - positivity at knot points.

    The optimization problem is solved with `quadprog`.

    TODO: Solve with `qpOASES` instead of `quadprog` to reduce
    dependency.

    """
    N = ss.shape[0] - 1
    x_p = np.linspace(ss[0], ss[-1], m + 1)  # Break points

    # Variable to the optimization problem, C, has n_col elements Each
    # piece of polynomial has k + 1 coefficients, corresponding to the
    # monomial.
    n_col = (k + 1) * m

    G = np.zeros((N + 1, n_col))
    for i in range(N + 1):
        found_bp = False
        for bp in range(m):
            if x_p[bp] <= ss[i] and ss[i] <= x_p[bp + 1]:
                found_bp = True
                break
        assert found_bp, "Breakpoint not found. Terminating"
        row = [(ss[i] - x_p[bp]) ** (k - j) for j in range (k + 1)]
        G[i, bp * (k + 1): (bp + 1) * (k + 1)] = row

    # End-points : g_0.T C = x_0; g_N.T C = x_N
    g_0 = np.zeros(n_col)
    g_0[0: k + 1] = [(ss[0] - x_p[0]) ** (k - j) for j in range(k + 1)]
    g_N = np.zeros(n_col)
    g_N[(m - 1) * (k + 1):] = [(ss[-1] - x_p[-2]) ** (k - j) for j in range(k + 1)]

    # C0 Continuity : HC = 0
    H0 = np.zeros((m - 1, n_col))
    for i in range(1, m):
        row_lhs = [(x_p[i] - x_p[i - 1]) ** (k - j) for j in range(k + 1)]
        row_rhs = [- (x_p[i] - x_p[i]) ** (k - j) for j in range(k + 1)]
        H0[i - 1, (i - 1) * (k + 1): i * (k + 1)] = row_lhs
        H0[i - 1, i * (k + 1): (i + 1) * (k + 1)] = row_rhs

    # C1 Continuity : HC = 0
    H1 = np.zeros((m - 1, n_col))
    for i in range(1, m):
        row_lhs = [(k - j) * (x_p[i] - x_p[i - 1]) ** (k - j - 1) for j in range(k)] + [0]
        row_rhs = [- (k - j) * (x_p[i] - x_p[i]) ** (k - j - 1) for j in range(k)] + [0]
        H1[i - 1, (i - 1) * (k + 1): i * (k + 1)] = row_lhs
        H1[i - 1, i * (k + 1): (i + 1) * (k + 1)] = row_rhs

    # Form a quadratic problem and solve with quadprog
    C_qp = np.vstack((g_0.reshape(1, -1), g_N.reshape(1, -1), H0, H1, G)).T
    b_qp = np.r_[sds[0], sds[N],
                 np.zeros(H0.shape[0]),
                 np.zeros(H1.shape[0]),
                 np.zeros(G.shape[0])]
    meq_qp = 1 + 1 + H0.shape[0] + H1.shape[0]
    G_qp = G.T.dot(G) + np.eye(n_col) * eps
    a_qp = sds.dot(G)
    res = quadprog.solve_qp(G_qp, a_qp, C_qp, b_qp, meq_qp)

    # Verify with scipy.PPoly
    C = res[0]
    pp = PPoly(C.reshape(m, k + 1).T, x_p)

    return pp
