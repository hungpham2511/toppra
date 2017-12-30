import numpy as np
import quadprog
from scipy.interpolate import PPoly


def compute_trajectory_gridpoints(path, sgrid, ugrid, xgrid):
    """ Compute a trajectory sampled at gridpoints.

    Parameters
    ----------
    path : :class:`.SplineInterpolator`
        The geometric path to parametrize. Can also be
        :class:`.PolynomialInterpolator` or
        :class:`.UnivariateSplineInterpolator`.
    sgrid : array
        Shape (N+1,). Array of gridpoints.
    ugrid : array
        Shape (N,). Array of controls.
    xgrid : array
        Shape (N+1,). Array of squared velocities.

    Returns
    -------
    tgrid : array
        Shape (N+1). Time at each gridpoints.
    q : array
        Shape (N+1, dof). Joint positions at each gridpoints.
    qd : array
        Shape (N+1, dof). Joint velocities at each gridpoints.
    qdd : array
        Shape (N+1, dof). Joint accelerations at each gridpoints.
    """
    tgrid = np.zeros_like(sgrid)
    N = sgrid.shape[0] - 1
    sdgrid = np.sqrt(xgrid)
    for i in range(N):
        tgrid[i + 1] = ((sgrid[i + 1] - sgrid[i]) / (sdgrid[i] + sdgrid[i + 1]) * 2
                        + tgrid[i])
    sddgrid = np.hstack((ugrid, ugrid[-1]))
    q = path.eval(sgrid)
    qs = path.evald(sgrid)  # derivative w.r.t [path position] s
    qss = path.evaldd(sgrid)
    array_mul = lambda v_arr, s_arr: np.array(
        [v_arr[i] * s_arr[i] for i in range(N + 1)])
    qd = array_mul(qs, sdgrid)
    qdd = array_mul(qs, sddgrid) + array_mul(qss, sdgrid ** 2)
    return tgrid, q, qd, qdd


def compute_trajectory_uniform(path, sgrid, ugrid, xgrid, dt=1e-2, smooth=False, smooth_eps=1e-4):
    """Compute trajectory with uniform sampling time.

    Notes
    -----
    
    If `smooth` is True, the return trajectory is smoothed with
    least-square technique. The return trajectory is guaranteed to
    satisfy the discrete transition relation. That is

    .. math::

        \mathbf{q}[i+1] & = \mathbf{q}[i] + \dot{\mathbf{q}}[i] dt + \ddot{\mathbf{q}}[i] dt ^ 2 / 2 \\\\
        \dot{\mathbf{q}}[i+1] & = \dot{\mathbf{q}}[i] + \ddot{\mathbf{q}}[i] dt

    while minimizing the difference with the original non-smooth
    trajectory computed with TOPP.

    If one finds that the function takes too much time to terminate,
    then it is very likely that the most time-consuming part is
    least-square. In this case, there are several options that one
    might take.

    1. Set `smooth` to False. This might return badly conditioned trajectory.
    2. Reduce `dt`. This is the recommended option.

    TODO: Implement a better *and more importantly, faster,* method to
    smoothing.

    Parameters
    ----------
    path : :class:`.SplineInterpolator`
        The geometric path to parametrize. Can also be
        :class:`.PolynomialInterpolator` or
        :class:`.UnivariateSplineInterpolator`.
    sgrid : array
        Shape (N+1,). Grid points.
    ugrid : array
        Shape (N,). Controls.
    xgrid : array
        Shape (N+1,). Squared velocities.
    dt : float, optional
        Sampling time step.
    smooth : bool, optional
        If True, do least-square smoothing. See note for more details.
    smooth_eps : float, optional
        Relative gain of minimizing variations of joint accelerations.

    Returns
    -------
    tgrid : array
        Shape (M). Time at each gridpoints.
    q : array
        Shape (M, dof). Joint positions at each gridpoints.
    qd : array
        Shape (M, dof). Joint velocities at each gridpoints.
    qdd : array
        Shape (M, dof). Joint accelerations at each gridpoints.

    """
    tgrid = np.zeros_like(sgrid)  # Array of time at each gridpoint
    N = sgrid.shape[0] - 1
    sdgrid = np.sqrt(xgrid)
    for i in range(N):
        tgrid[i + 1] = ((sgrid[i + 1] - sgrid[i]) / (sdgrid[i] + sdgrid[i + 1]) * 2
                        + tgrid[i])
    # shape (M+1,) array of sampled time
    tsample = np.arange(tgrid[0], tgrid[-1], dt)
    ssample = np.zeros_like(tsample)  # sampled position
    xsample = np.zeros_like(tsample)  # sampled velocity squared
    sdsample = np.zeros_like(tsample)  # sampled velocity
    usample = np.zeros_like(tsample)  # sampled path acceleration
    igrid = 0
    for i, t in enumerate(tsample):
        while t > tgrid[igrid + 1]:
            igrid += 1
        usample[i] = ugrid[igrid]
        sdsample[i] = sdgrid[igrid] + (t - tgrid[igrid]) * usample[i]
        xsample[i] = sdsample[i] ** 2
        ssample[i] = (sgrid[igrid] +
                      (xsample[i] - xgrid[igrid]) / 2 / usample[i])

    q = path.eval(ssample)
    qs = path.evald(ssample)  # derivative w.r.t [path position] s
    qss = path.evaldd(ssample)

    def array_mul(vectors, scalars):
        # given array of vectors and array of scalars
        # multiply each vector with each scalar
        res = np.zeros_like(vectors)
        for i in range(scalars.shape[0]):
            res[i] = vectors[i] * scalars[i]
        return res

    qd = array_mul(qs, sdsample)
    qdd = array_mul(qs, usample) + array_mul(qss, sdsample ** 2)

    if not smooth:
        return tsample, q, qd, qdd
    else:
        # Least square smoothing
        logger.debug("Compute trajectory with least-square smoothing.")
        dof = q.shape[1]
        A = np.array([[1., dt], [0, 1.]])
        B = np.array([dt ** 2 / 2, dt])
        M = tsample.shape[0] - 1
        Phi = np.zeros((2 * M, M))
        for i in range(M):  # Block diagonal
            Phi[2 * i: 2 * i + 2, i] = B
        for i in range(1, M):  # First column
            Phi[2 * i: 2 * i + 2, 0] = np.dot(A, Phi[2 * i - 2: 2 * i, 0])
        for i in range(1, M):  # Next column
            Phi[2 * i:, i] = Phi[2 * i - 2: 2 * M - 2, i - 1]

        Beta = np.zeros((2 * M, 2))
        Beta[0: 2, :] = A
        for i in range(1, M):
            Beta[2 * i: 2 * i + 2, :] = np.dot(A, Beta[2 * i - 2: 2 * i, :])

        Delta = np.zeros((M - 1, M))
        for i in range(M - 1):
            Delta[i, i] = 1
            Delta[i, i + 1] = - 1

        for k in range(dof):
            Xd = np.vstack((q[1:, k], qd[1:, k])).T.flatten()  # numpy magic
            x0 = np.r_[q[0, k], qd[0, k]]
            xM = np.r_[q[-1, k], qd[-1, k]]

            G = np.dot(Phi.T, Phi) + np.dot(Delta.T, Delta) * smooth_eps
            a = - np.dot(Phi.T, Beta.dot(x0) - Xd)
            C = Phi[2 * M - 2:].T
            b = xM - Beta[2 * M - 2:].dot(x0)
            sol = quadprog.solve_qp(G, a, C, b, meq=2)[0]
            Xsol = np.dot(Phi, sol) + np.dot(Beta, x0)
            Xsol = Xsol.reshape(-1, 2)
            q[1:, k] = Xsol[:, 0]
            qd[1:, k] = Xsol[:, 1]
            qdd[:-1, k] = sol
            qdd[-1, k] = sol[-1]

        return tsample, q, qd, qdd


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
