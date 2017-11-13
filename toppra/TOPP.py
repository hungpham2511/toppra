"""This module constraints the main Path Parameterization solver.

"""

import numpy as np
from qpoases import (PyOptions as Options, PyPrintLevel as PrintLevel,
                     PyReturnValue as ReturnValue, PySQProblem as SQProblem)
import logging
import quadprog

logger = logging.getLogger(__name__)
SUCCESSFUL_RETURN = ReturnValue.SUCCESSFUL_RETURN


qpOASESReturnValueDict = {
    1: "SUCCESSFUL_RETURN",
    61: "HOTSTART_STOPPED_INFEASIBILITY",
    37: "INIT_FAILED_INFEASIBILITY "
}

# Constants
TINY = 1e-8
SMALL = 1e-5
INFTY = 1e8

###############################################################################
#                   PathParameterization Algorithms/Objects                   #
###############################################################################


def compute_trajectory_gridpoints(path, sgrid, ugrid, xgrid):
    """ Convert grid points to a trajectory.

    Parameters
    ----------
    path : interpolator
    sgrid : ndarray, shape (N+1,)
        Array of gridpoints.
    ugrid : ndarray, shape (N,)
        Array of controls.
    xgrid : ndarray, shape (N+1,)
        Array of squared velocities.

    Returns
    -------
    tgrid : ndarray, shape (N+1)
        Time at each gridpoints.
    q : ndarray, shape (N+1, dof)
        Joint positions at each gridpoints.
    qd : ndarray, shape (N+1, dof)
        Joint velocities at each gridpoints.
    qdd : ndarray, shape (N+1, dof)
        Joint accelerations at each gridpoints.
    """
    tgrid = np.zeros_like(sgrid)
    N = sgrid.shape[0] - 1
    sdgrid = np.sqrt(xgrid)
    for i in range(N):
        tgrid[i+1] = ((sgrid[i+1] - sgrid[i]) / (sdgrid[i] + sdgrid[i+1]) * 2
                      + tgrid[i])
    sddgrid = np.hstack((ugrid, ugrid[-1]))
    q = path.eval(sgrid)
    qs = path.evald(sgrid)  # derivative w.r.t [path position] s
    qss = path.evaldd(sgrid)
    array_mul = lambda v_arr, s_arr: np.array(
        [v_arr[i] * s_arr[i] for i in range(N+1)])
    qd = array_mul(qs, sdgrid)
    qdd = array_mul(qs, sddgrid) + array_mul(qss, sdgrid ** 2)
    return tgrid, q, qd, qdd


def compute_trajectory_points(path, sgrid,
                              ugrid, xgrid,
                              dt=1e-2, smooth=False,
                              smooth_eps=1e-4):
    """Compute trajectory with uniform sampling time.

    Note
    ----
    Additionally, if `smooth` is True, the return trajectory is smooth
    using least-square technique. The return trajectory, also
    satisfies the discrete transition relation. That is

    q[i+1] = q[i] + qd[i] * dt + qdd[i] * dt ^ 2 / 2
    qd[i+1] = qd[i] + qdd[i] * dt

    If one finds that the function takes too much time to terminate,
    then it is very likely that the most time-consuming part is
    least-square. In this case, there are several options that one
    might take.
    1. Set `smooth` to False. This might return badly conditioned
    trajectory.
    2. Reduce `dt`. This is the recommended option.


    Parameters
    ----------
    path : interpolator
    sgrid : ndarray, shape (N+1,)
        Array of gridpoints.
    ugrid : ndarray, shape (N,)
        Array of controls.
    xgrid : ndarray, shape (N+1,)
        Array of squared velocities.
    dt : float, optional
        Sampling time step.
    smooth : bool, optional
        If True, do least-square smoothing. See above for more details.
    smooth_eps : float, optional
        Relative gain of minimizing variations of joint accelerations.

    Returns
    -------
    tgrid : ndarray, shape (M)
        Time at each gridpoints.
    q : ndarray, shape (M, dof)
        Joint positions at each gridpoints.
    qd : ndarray, shape (M, dof)
        Joint velocities at each gridpoints.
    qdd : ndarray, shape (M, dof)
        Joint accelerations at each gridpoints.

    """
    tgrid = np.zeros_like(sgrid)  # Array of time at each gridpoint
    N = sgrid.shape[0] - 1
    sdgrid = np.sqrt(xgrid)
    for i in range(N):
        tgrid[i+1] = ((sgrid[i+1] - sgrid[i]) / (sdgrid[i] + sdgrid[i+1]) * 2
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
        logger.debug("Compute trajectory with least-square smoothing.")
        dof = q.shape[1]
        # Still slow, I will now try QP with quadprog
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
        for i in range(M-1):
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


class qpOASESPPSolver(object):
    """Parametrize geometric path using TOPP-RA.

    This class first translates a set of `PathConstraint` to the form
    needed by qpOASES form. Then it solve for the parameterization in
    two passes as described in the paper.

    Parameters
    ----------
    constraint_set : A List,
        A list of PathConstraint.

    Attributes
    ----------
    Ks : ndarray, shape (N+1, 2)
        Controllable sets/intervals.
    I0 : ndarray
    IN : ndarray
    ss : ndarray. Discretized path positions.
    nm : int
        Dimension of the canonical constraint part.
    nv : int
        Dimension of the combined slack.
    niq : int
        Dimension of the inequalities on slack.
    neq : int
        Dimension of the equalities on slack.
    nv : int
        Dimension of the (qpOASES) optimization variable.
    nC : int
        Dimension of the (qpOASES) constraint matrix.
    A : ndarray
        qpOASES coefficient matrix.
    lA : ndarray,
        qpOASES coefficient matrix.
    hA : ndarray
        qpOASES coefficient matrix.
    l : ndarray
        qpOASES coefficient matrix.
    h : ndarray
        qpOASES coefficient matrix.
    H : ndarray
        qpOASES coefficient matrix.
    g : ndarray
        qpOASES coefficient matrix.

    NOTE
    ----
    Note that the first `self.nop` row of A are all zeros. These are
    operational rows, used to specify additional constraints required
    for computing controllable/reachable sets.

    The coefficient matrices are to be input to the qpOASES solver as
    follow

            min     1/2 (u, x, v)^T H (u, x, v) + g^T (u, x, v)
            s.t.    lA <= A (u, x, v) <= hA
                    l  <=   (u, x, v) <= h

    More specifically, there are four sections in matrix A.


             | 0     | 0     |   0 |   0 |   0 |      Openration
             | 0     | 0     |   0 |   0 |   0 |      ----------
             |-------+-------+-----+-----+-----+
             | a1    | b1    |   0 |   0 |   0 |      Canonical
             | a2    | b2    |   0 |   0 |   0 |      ---------
             | a3    | b3    |   0 |   0 |   0 |
        A[i]=|-------+-------+-----+-----+-----+
             | abar1 | bbar1 | -D1 |   0 |   0 |      Equality
             | abar2 | bbar2 |   0 | -D2 |   0 |      --------
             | abar3 | bbar3 |   0 |   0 | -D3 |
             |-------+-------+-----+-----+-----+
             | 0     | 0     |  G1 |     |     |      Inequality
             | 0     | 0     |     |  G2 |     |      ----------
             | 0     | 0     |     |     |  G3 |

    And correspondingly, four sections in matrices lA[i], hA[i]

             | 0      |         | 0      |      Openration
             | 0      |         | 0      |      ----------
             |--------+         |--------+
             | -infty |         | 0      |      Canonical
             | -infty |         | 0      |      ---------
             | -infty |         | 0      |
       lA[i]=|--------+    hA = |--------+
             | -cbar1 |         | -cbar1 |      Equality
             | -cbar2 |         | -cbar2 |      --------
             | -cbar3 |         | -cbar3 |
             |--------+         |--------+
             | lG1    |         | hG1    |      Inequality
             | lG2    |         | hG2    |      ----------
             | lG3    |         | hG3    |

    Finally, bound vectors l, h are given as:

             | -INFTY | for u  |       | INFTY | for u  |
             | 0.     | for x  |       | INFTY | for x  |
        l[i]=| l1     | for v1 |, h[i]=| h1    | for v1 |
             | l2     |        |       | h2    |        |
             | l3     |        |       | h3    |        |
    """
    def __init__(self, constraint_set, verbose=False):
        self.I0 = np.r_[0, 1e-4]  # Start and end velocity interval
        self.IN = np.r_[0, 1e-4]
        self.ss = constraint_set[0].ss
        self.Ds = self.ss[1:] - self.ss[:-1]
        self.N = constraint_set[0].N
        for c in constraint_set:
            assert np.allclose(c.ss, self.ss)

        # Controllable subsets
        self._K = - np.ones((self.N + 1, 2))
        self._L = - np.ones((self.N + 1, 2))
        self.nop = 3  # Operational row, used for special constraints

        self.constraint_set = constraint_set

        # Pre-processing: Compute shape and init zero coeff matrices
        self._init_matrices(constraint_set)
        self._fill_matrices()
        summary_msg = """
Initialize Path Parameterization instance
------------------------------
\t N                  : {:8d}
\t No. of constraints : {:8d}
\t No. of slack var   : {:8d}
\t No. of can. ineq.  : {:8d}
\t No. of equalities  : {:8d}
\t No. of inequalities: {:8d}
""".format(self.N, len(self.constraint_set),
           self.nv, self. nm, self.neq, self.niq)
        logger.info(summary_msg)

        # Setup solvers
        self._init_qpoaes_solvers(verbose)

    @property
    def K(self):
        """ The Controllable subsets.
        """
        controllable_subsets = self._K[:, 0] > - TINY
        return self._K[controllable_subsets]

    @property
    def L(self):
        """ The Reachable subsets.
        """
        reachable_subsets = self._L[:, 0] > - TINY
        return self._L[reachable_subsets]

    def set_start_interval(self, I0):
        """ Set the starting squared velocity interval.

        Parameters
        ----------
        I0: (2, ) float tuple.
            Starting (x_lower, x_higher) squared path velocities.
        """
        self.I0 = I0

    def set_goal_interval(self, IN):
        """ Set the goal squared velocity interval.

        Parameters
        ----------
        IN: (2, ) float tuple.
            Starting (x_lower, x_higher) squared path velocities.
        """
        self.IN = IN

    def _init_qpoaes_solvers(self, verbose):
        """Setup two qpoases solvers following warm-up strategy.
        """
        # Max Number of working set change.
        self.nWSR_cnst = 1000
        _, nC, nV = self.A.shape
        # Setup solver
        options = Options()
        if verbose:
            logger.debug("Set qpOASES print level to HIGH")
            options.printLevel = PrintLevel.HIGH
        else:
            logger.debug("Set qpOASES print level to NONE")
            options.printLevel = PrintLevel.NONE
        self.solver_up = SQProblem(nV, nC)
        self.solver_up.setOptions(options)
        self.solver_down = SQProblem(nV, nC)
        self.solver_down.setOptions(options)

    def _init_matrices(self, constraint_set):
        """ Initialize coefficient matrices for qpOASES.
        """
        self.nm = sum([c.nm for c in constraint_set])
        self.niq = sum([c.niq for c in constraint_set])
        self.neq = sum([c.neq for c in constraint_set])
        self.nv = sum([c.nv for c in constraint_set])
        self.nV = self.nv + 2
        self.nC = self.nop + self.nm + self.neq + self.niq

        self.H = np.zeros((self.nV, self.nV))
        self.g = np.zeros(self.nV)
        # fixed bounds
        self.l = np.zeros((self.N+1, self.nV))
        self.h = np.zeros((self.N+1, self.nV))
        # lA, A, hA constraints
        self.lA = np.zeros((self.N+1, self.nC))
        self.hA = np.zeros((self.N+1, self.nC))
        self.A = np.zeros((self.N+1, self.nC, self.nV))
        self._xfull = np.zeros(self.nV)  # interval vector, store primal
        self._yfull = np.zeros(self.nC)  # interval vector, store dual

    def _fill_matrices(self):
        """Fill coefficient matrices.

        See the qpOASESPPSolver's class description for more details
        regarding the matrices.

        Args:
        ----
        i: An Int. An positive index, less than or equal to N.

        """
        self.g.fill(0)
        self.H.fill(0)
        # A
        self.A.fill(0)
        self.A[:, :self.nop, :] = 0.  # operational rows
        self.lA[:, :self.nop] = 0.
        self.hA[:, :self.nop] = 0.
        # canonical
        row = self.nop
        for c in filter(lambda c: c.nm != 0, self.constraint_set):
            self.A[:, row: row + c.nm, 0] = c.a
            self.A[:, row: row + c.nm, 1] = c.b
            self.lA[:, row: row + c.nm] = - INFTY
            self.hA[:, row: row + c.nm] = - c.c
            row += c.nm

        # equalities
        row = self.nop + self.nm
        col = 2
        for c in filter(lambda c: c.neq != 0, self.constraint_set):
            self.A[:, row: row + c.neq, 0] = c.abar
            self.A[:, row: row + c.neq, 1] = c.bbar
            self.A[:, row: row + c.neq, col: col + c.nv] = - c.D
            self.lA[:, row: row + c.neq] = - c.cbar
            self.hA[:, row: row + c.neq] = - c.cbar
            row += c.neq
            col += c.nv

        # inequalities
        row = self.nop + self.nm + self.neq
        col = 2
        for c in filter(lambda c: c.niq != 0, self.constraint_set):
            self.A[:, row: row + c.niq, col: col + c.nv] = c.G
            self.lA[:, row: row + c.niq] = c.lG
            self.hA[:, row: row + c.niq] = c.hG
            row += c.niq
            col += c.nv

        # bounds on var
        self.l[:, 0] = - INFTY  # - infty <= u <= infty
        self.h[:, 0] = INFTY
        self.l[:, 1] = 0  # 0 <= x <= infty
        self.h[:, 1] = INFTY
        row = 2
        for c in filter(lambda c: c.nv != 0, self.constraint_set):
            self.l[:, row: row + c.nv] = c.l
            self.h[:, row: row + c.nv] = c.h
            row += c.nv

    def solve_controllable_sets(self, eps=1e-8):
        """Solve for controllable sets K(i, IN).

        The i-th controllable set K(i, IN) is the set of states at
        s=s_i for which there exists at least a sequence of admissible
        controls that drives it to IN.

        Parameters
        ----------
        eps: float, optional
             The safety margin guarded againts numerical error. This
             number needs not be too high.

        Returns
        -------
        out: bool.
             True if K(0, IN) is not empty.

        """
        self._reset_operational_matrices()
        self.nWSR_up = np.ones((self.N+1, 1), dtype=int) * self.nWSR_cnst
        self.nWSR_down = np.ones((self.N+1, 1), dtype=int) * self.nWSR_cnst
        xmin, xmax = self.proj_x_admissible(self.N, self.IN[0],
                                            self.IN[1], init=True)
        if xmin is None:
            logger.warn("Fail to project the interval IN to feasibility")
            return False
        else:
            self._K[self.N, 1] = xmax
            self._K[self.N, 0] = xmin

        init = True
        for i in range(self.N - 1, -1, -1):
            xmin_i, xmax_i = self.one_step(
                i, self._K[i + 1, 0], self._K[i + 1, 1], init=init)
            # Turn init off, use hotstart
            init = False
            if xmin_i is None:
                logger.warn("Find controllable set K(%d, IN failed", i)
                return False
            else:
                self._K[i, 1] = xmax_i - eps  # Buffer for numerical error
                self._K[i, 0] = xmin_i + eps

        return True

    def solve_reachable_sets(self):
        """Solve for reachable sets L(i, I0).

        """
        self._reset_operational_matrices()
        xmin, xmax = self.proj_x_admissible(
            0, self.I0[0], self.I0[1], init=True)
        if xmin is None:
            logger.warn("Fail to project the interval I0 to feasibility")
            return False
        else:
            self._L[0, 1] = xmax
            self._L[0, 0] = xmin
        for i in range(self.N):
            init = (True if i <= 1 else False)
            xmin_nx, xmax_nx = self.reach(i, self._L[i, 0], self._L[i, 1], init=init)
            if xmin_nx is None:
                logger.warn("Forward propagation from L%d failed ", i)
                return False
            xmin_pr, xmax_pr = self.proj_x_admissible(i + 1, xmin_nx, xmax_nx, init=init)
            if xmin_pr is None:
                logger.warn("Projection for L{:d} failed".format(i))
                return False
            else:
                self._L[i + 1, 1] = xmax_pr
                self._L[i + 1, 0] = xmin_pr
        return True

    def solve_topp(self, save_solutions=False, reg=0.):
        """Solve for the time-optimal path-parameterization

        Parameters
        ----------
        save_solutions: bool
            Save solutions of each step.
        reg: float
            Regularization gain.

        Returns
        -------
        us: ndarray, shape (N,)
            Contains the TOPP's controls.
        xs: ndarray, shape (N+1,)
            Contains the TOPP's squared velocities.

        """
        if save_solutions:
            self._xfulls = np.empty((self.N, self.nV))
            self._yfulls = np.empty((self.N, self.nC))
        # Backward pass
        controllable = self.solve_controllable_sets()
        # Check controllability
        infeasible = (self._K[0, 1] < self.I0[0] or self._K[0, 0] > self.I0[1])

        if not controllable or infeasible:
            msg = """
Unable to parameterizes this path:
- K(0) is empty : {0}
- sd_start not in K(0) : {1}
""".format(controllable, infeasible)
            raise ValueError(msg)

        # Forward pass
        # Setup matrices
        self._reset_operational_matrices()
        # Enforce x == xs[i]
        self.A[:, 0, 1] = 1.
        self.A[:, 0, 0] = 0.
        # Enfore Kmin <= x + 2 ds u <= Kmax
        self.A[:, 1, 1] = 1.
        self.A[:self.N, 1, 0] = 2 * self.Ds
        self.nWSR_topp = np.ones((self.N+1, 1), dtype=int) * self.nWSR_cnst
        # Setup matrices finished
        xs = np.zeros(self.N + 1)
        us = np.zeros(self.N)
        xs[0] = min(self._K[0, 1], self.I0[1])
        _, _ = self.topp_step(0, xs[0], self._K[1, 0], self._K[1, 1],
                              init=True, reg=reg)  # Warm start
        for i in range(self.N):
            u_, x_ = self.topp_step(i, xs[i], self._K[i+1, 0], self._K[i+1, 1],
                                    init=False, reg=reg)
            xs[i+1] = x_
            us[i] = u_
            if save_solutions:
                self._xfulls[i] = self._xfull.copy()
                # self._yfulls[i] = self._yfull.copy()
        return us, xs

    @property
    def slack_vars(self):
        """ Recent stored slack variable.
        """
        return self._xfulls[:, 2:]

    def _reset_operational_matrices(self):
        """Reset the coefficient matrix.

        It is important to use this function whenever starting a new
        operation. For instance, computing the MVC.

        """
        # reset all rows
        self.A[:, :self.nop] = 0
        self.lA[:, :self.nop] = 0
        self.hA[:, :self.nop] = 0
        self.H[:, :] = 0
        self.g[:] = 0

    ###########################################################################
    #                    Main Set Projection Functions                        #
    ###########################################################################
    def one_step(self, i, xmin, xmax, init=False):
        """Compute the one-step set for the interval [xmin, xmax]
        
        Parameters
        ----------
        i: int
        xmin: float
        xmax: float
        init: bool, optional
            Use qpOASES with hotstart.

        Returns
        -------
        xmin_i: float
        xmax_i: float

        Definition:
        ----------

            The one-step set $\calQ(i, \bbI)$ is the largest set of
            states in stage $i$ for which there exists an admissible
            control that steers the system to a state in $\bbI$.

        NOTE:
        - self.nWSR_up and self.nWSR_down need to be initialized prior.
        - If the projection is not feasible (for example when xmin >
        xmax), then return None, None.

        """
        # Set constraint: xmin <= 2 ds u + x <= xmax
        self.A[i, 0, 1] = 1
        self.A[i, 0, 0] = 2 * (self.ss[i + 1] - self.ss[i])
        self.lA[i, 0] = xmin
        self.hA[i, 0] = xmax

        if init:
            # upper solver solves for max x
            self.g[1] = -1.
            res_up = self.solver_up.init(
                self.H, self.g, self.A[i], self.l[i], self.h[i], self.lA[i],
                self.hA[i], self.nWSR_up[i])

            # lower solver solves for min x
            self.g[1] = 1.
            res_down = self.solver_down.init(
                self.H, self.g, self.A[i], self.l[i], self.h[i], self.lA[i],
                self.hA[i], self.nWSR_down[i])
        else:
            # upper solver solves for max x
            self.g[1] = -1.
            res_up = self.solver_up.hotstart(
                self.H, self.g, self.A[i], self.l[i], self.h[i], self.lA[i],
                self.hA[i], self.nWSR_up[i])

            # lower bound
            self.g[1] = 1.
            res_down = self.solver_down.hotstart(
                self.H, self.g, self.A[i], self.l[i], self.h[i],
                self.lA[i], self.hA[i], self.nWSR_down[i])

        # Check result
        if (res_up != SUCCESSFUL_RETURN) or (res_down != SUCCESSFUL_RETURN):
            logger.warn("""
Computing one-step failed.

    INFO:
    ----
        i                     = {}
        xmin                  = {}
        xmax                  = {}
        warm_start            = {}
        upper LP solve status = {}
        lower LP solve status = {}
""".format(i, xmin, xmax, init, res_up, res_down))

            return None, None

        # extract solution
        self.solver_up.getPrimalSolution(self._xfull)
        xmax_i = self._xfull[1]
        self.solver_down.getPrimalSolution(self._xfull)
        xmin_i = self._xfull[1]
        return xmin_i, xmax_i

    def reach(self, i, xmin, xmax, init=False):
        """Compute the reach set from [xmin, xmax] at stage i.

        Parameters
        ----------
        i: int
        xmin: float
        xmax: float
        init: bool, optional
            Use qpOASES with hotstart.

        Returns
        -------
        xmin_i: float
        xmax_i: float

        Definition:
        -----------

            The reach set $\calR(i, \bbI)$ is the set of states in $\bbR$
            to which the system will evolve given any admissible states
            in $\bbI$ and any admissible controls in the $i$-th stage.


        If the projection is not feasible (for example when xmin >
        xmax), then return None, None.
        """

        self.A[i, 0, 1] = 1
        self.A[i, 0, 0] = 0.
        self.lA[i, 0] = xmin
        self.hA[i, 0] = xmax

        # upper bound
        nWSR_up = np.array([self.nWSR_cnst])
        self.g[0] = -2. * (self.ss[i + 1] - self.ss[i])
        self.g[1] = -1.
        if init:
            res_up = self.solver_up.init(
                self.H, self.g, self.A[i], self.l[i], self.h[i], self.lA[i],
                self.hA[i], nWSR_up)
        else:
            res_up = self.solver_up.hotstart(
                self.H, self.g, self.A[i], self.l[i], self.h[i], self.lA[i],
                self.hA[i], nWSR_up)

        nWSR_down = np.array([self.nWSR_cnst])
        self.g[0] = 2. * (self.ss[i + 1] - self.ss[i])
        self.g[1] = 1.
        if init:
            res_down = self.solver_down.init(
                self.H, self.g, self.A[i], self.l[i], self.h[i], self.lA[i],
                self.hA[i], nWSR_down)
        else:
            res_down = self.solver_down.hotstart(
                self.H, self.g, self.A[i], self.l[i], self.h[i], self.lA[i],
                self.hA[i], nWSR_down)

        if (res_up != SUCCESSFUL_RETURN) or (res_down != SUCCESSFUL_RETURN):
            logger.warn("""
Computing reach set failed.

    INFO:
    ----
        i                     = {}
        xmin                  = {}
        xmax                  = {}
        warm_start            = {}
        upper LP solve status = {}
        lower LP solve status = {}
""".format(i, xmin, xmax, init, res_up, res_down))
            return None, None

        # extract solution
        xmax_i = -self.solver_up.getObjVal()
        xmin_i = self.solver_down.getObjVal()
        return xmin_i, xmax_i

    def proj_x_admissible(self, i, xmin, xmax, init=False):
        """Project the interval [xmin, xmax] back to the feasible set.

        Parameters
        ----------
        i: int
           Index of the path position where the project is at.
        xmin: float
            Lower bound of the interval to be projected
        xmax: float
            Upper bound of the interval to be projected
        init: bool, optional
            If True, use qpOASES without hotstart.
            If False, use hotstart.

        Returns
        -------
        xmin_i: float
        xmax_i: float

        Notes
        -----
        If the projection is infeasible, for example when xmin > xmax
        or when the infeasible set if empty, then return None, None.

        If one find unreasonable results such as (0, 0) constantly
        appear, it is a good ideal to try reseting the internal
        operational matrices using this function

            self._reset_operational_matrices()

        One should now see correct results appearing.
        """

        self.A[i, 0, 1] = 1
        self.A[i, 0, 0] = 0.
        self.lA[i, 0] = xmin
        self.hA[i, 0] = xmax

        # upper bound
        nWSR_up = np.array([self.nWSR_cnst])
        self.g[0] = 0.
        self.g[1] = -1.
        if init:
            res_up = self.solver_up.init(
                self.H, self.g, self.A[i], self.l[i], self.h[i], self.lA[i],
                self.hA[i], nWSR_up)
        else:
            res_up = self.solver_up.hotstart(
                self.H, self.g, self.A[i], self.l[i], self.h[i], self.lA[i],
                self.hA[i], nWSR_up)

        nWSR_down = np.array([self.nWSR_cnst])
        self.g[0] = 0.
        self.g[1] = 1.
        if init:
            res_down = self.solver_down.init(
                self.H, self.g, self.A[i], self.l[i], self.h[i], self.lA[i],
                self.hA[i], nWSR_down)
        else:
            res_down = self.solver_down.hotstart(
                self.H, self.g, self.A[i], self.l[i], self.h[i], self.lA[i],
                self.hA[i], nWSR_down)

        if (res_up != SUCCESSFUL_RETURN) or (res_down != SUCCESSFUL_RETURN):
            logger.warn("""
Computing projection failed.

    INFO:
    ----
        i                     = {}
        xmin                  = {}
        xmax                  = {}
        warm_start            = {}
        upper LP solve status = {}
        lower LP solve status = {}
""".format(i, xmin, xmax, init, res_up, res_down))
            logger.warn(
                "qpOASES error code {:d} is {}".format(
                    res_up,
                    qpOASESReturnValueDict[res_up]))
            return None, None

        # extract solution
        self.solver_up.getPrimalSolution(self._xfull)
        xmax_i = self._xfull[1]
        self.solver_down.getPrimalSolution(self._xfull)
        xmin_i = self._xfull[1]
        assert xmin_i <= xmax_i, "Numerical error inside `proj_x_admissible`."
        return xmin_i, xmax_i

    def topp_step(self, i, x, xmin, xmax, init=False, reg=0.):
        """ Find max u such that xmin <= x + 2 ds u <= xmax.

        If the projection is not feasible (for example when xmin >
        xmax), then return None, None.

        NOTE:
        - self.nWSR_topp need to be initialized prior.
        """
        # Constraint 1: x = x
        self.lA[i, 0] = x
        self.hA[i, 0] = x
        # Constraint 2: xmin <= 2 ds u + x <= xmax
        self.lA[i, 1] = xmin
        self.hA[i, 1] = xmax

        # Objective
        # max  u + reg ||v||_2^2
        self.g[0] = -1.
        if self.nv != 0:
            self.H[2:, 2:] += np.eye(self.nv) * reg

        if init:
            res_up = self.solver_up.init(
                self.H, self.g, self.A[i], self.l[i], self.h[i], self.lA[i],
                self.hA[i], self.nWSR_topp[i])
        else:
            res_up = self.solver_up.hotstart(
                self.H, self.g, self.A[i], self.l[i], self.h[i], self.lA[i],
                self.hA[i], self.nWSR_topp[i])

        if (res_up != SUCCESSFUL_RETURN):
            logger.warn("Non-optimal solution at i=%d. Returning default.", i)
            return None, None

        # extract solution
        self.solver_up.getPrimalSolution(self._xfull)
        # self.solver_up.getDualSolution(self._yfull)  # cause failure
        u_topp = self._xfull[0]
        return u_topp, x + 2 * self.Ds[i] * u_topp
