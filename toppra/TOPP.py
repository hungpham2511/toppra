from enum import Enum
import numpy as np
from qpoases import (PyOptions as Options, PyPrintLevel as PrintLevel,
                     PyReturnValue as ReturnValue, PySQProblem as SQProblem)

from rave import Rave
from rave.Rave import compute_jacobian_wrench
from scipy.linalg import block_diag
from _CythonUtils import _create_velocity_constraint

SUCCESSFUL_RETURN = ReturnValue.SUCCESSFUL_RETURN


# Paper directory
def paper_dir():
    return "/home/hung/git/hung/Papers/2016-TOPP-Stochastic/figures/"


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
#               PathConstraint and constraint handling functions              #
###############################################################################
class PathConstraintKind(Enum):
    Canonical = 0
    TypeI = 1
    TypeII = 2


class PathConstraint(object):
    """Discretized constraint on a path.

    Generally, at position s, we consider the following constraints:

                a(s) u    + b(s) x    + c(s)    <= 0
                abar(s) u + bbar(s) x + cbar(s)  = D(s) v

                lG(s) <=     G(s) v             <= hG(s)
                l(s)  <=          v             <= h(s)

    where u is the acceleration, x is the squared velocity, v a slack
    variable whose physical meanings depend on the nature of the
    constraint.

    The PathConstraint object can represent both the collocated
    constraints and the interpolated constraints. Note that the
    factory functions `create_[..]_path_constraint` only return the
    collocated version. To create a interpolated PathConstraint, one
    needs to use the function `interpolate_constraint`.

    The Attribute `ss` stores the sampled positions.

    Attributes:
    -----------
      _nm : An Int. Dimension of non-redundant inequalities.
      _nv : An Int. Dimension of slack variable.
      _neq : An Int. Dimension of equality constraint
      _niq : An Int. Dimension of inequality constraint

      a, b, c: Three ndarray(s). Shaped (N+1, nm).
      abar, bbar, cbar: Three ndarray(s). Shaped (N+1, neq).
      D: An ndarray. Shaped (N+1, neq, nv)
      lG, hG: Two ndarray(s). Shaped (N+1, niq).
      l, h: Two ndarray(s). Shaped (N+1, nv).
      G: A ndarray. Shaped (N+1, nv, neq)
    """

    def __repr__(self):
        return "Constraint(nm:{:d}, neq:{:d}, nv:{:d}, niq:{:d})".format(
            self.nm, self.neq, self.nv, self.niq)

    def __lt__(self, pc):
        return self.kind.value < pc.kind.value

    def __init__(self, a=None, b=None, c=None,
                 abar=None, bbar=None, cbar=None,
                 D=None, l=None, h=None,
                 lG=None, G=None, hG=None,
                 name=None, ss=None):
        """Initizale with coefficient matrices.

        A matrix if None will be set to an empty ndarray.

        Args:
        -----
        a     : An (N+1, neq)     ndarray.
                        See above for definition.
        b     : An (N+1, neq)     ndarray.
        c     : An (N+1, neq)     ndarray.
        abar  : An (N+1, neq)     ndarray.
                        See above for definition.
        bbar  : An (N+1, neq)     ndarray.
        cbar  : An (N+1, neq)     ndarray.
        lG    : An (N+1, niq)     ndarray or None.
        hG    : An (N+1, niq)     ndarray or None.
        G     : An (N+1, niq, nv) ndarray or None.
        l     : An (N+1, nv)      ndarray or None.
        h     : An (N+1, nv)      ndarray or None.
        name : A String. Represents the name of the constraint.
        """
        self.N = ss.shape[0] - 1  # number of intervals
        # store constraints matrices
        if a is None:
            self.a = np.empty((self.N + 1, 0))
            self.b = np.empty((self.N + 1, 0))
            self.c = np.empty((self.N + 1, 0))
        else:
            self.a = a
            self.b = b
            self.c = c
        self._nm = self.a.shape[1]

        if D is None:
            self.abar = np.empty((self.N + 1, 0))
            self.bbar = np.empty((self.N + 1, 0))
            self.cbar = np.empty((self.N + 1, 0))
            self.D = np.empty((self.N + 1, 0, 0))
        else:
            self.abar = abar
            self.bbar = bbar
            self.cbar = cbar
            self.D = D
        self._neq = self.abar.shape[1]
        self._nv = self.D[0].shape[1]

        if l is None:
            self.l = np.empty((self.N + 1, 0))
            self.h = np.empty((self.N + 1, 0))
        else:
            self.l = l
            self.h = h

        if lG is None:
            self.lG = np.empty((self.N + 1, 0))
            self.G = np.empty((self.N + 1, 0, self.nv))
            self.hG = np.empty((self.N + 1, 0))
        else:
            self.lG = lG
            self.G = G
            self.hG = hG
        self._niq = self.lG.shape[1]

        self.name = name
        self._ss = ss

        # Assert kind
        if self.nm != 0:
            self._kind = PathConstraintKind.Canonical
        elif self.niq == 0:
            self._kind = PathConstraintKind.TypeI
        else:
            self._kind = PathConstraintKind.TypeII

    @property
    def kind(self):
        """ The kind of Path Constraint.
        """
        return self._kind

    @property
    def ss(self):
        """ Discretization grid points.
        """
        return self._ss

    @property
    def nm(self):
        """Dimension of canonical constraints (inequalities)

        A canonical constraint has this form:

              a[i] * u + b[i] * x + c[i] <= 0,

        where a[i] is a `nm`-dimensional vector.
        """
        return self._nm

    @property
    def neq(self):
        """Dimension of redundant constraints (equalities).
        """
        return self._neq

    @property
    def niq(self):
        """Dimension (number of) inequalities on the slack variable v.
        """
        return self._niq

    @property
    def nv(self):
        """Dimension (number of) of the slack variable.
        """
        return self._nv


def interpolate_constraint(pc):
    """Produce a first-order interpolation discrete constraint.

    Args:
         pc: A PathConstraint. This is the original collocation constraint.

    Returns:
         pc_intp: A PathConstraint. The interpolated constraint.
    """
    N = pc.N
    Ds = pc.ss[1:] - pc.ss[:N]
    # Canonical part
    a_intp = np.empty((pc.N+1, 2 * pc._nm))
    a_intp[:, 0:pc._nm] = pc.a
    a_intp[N] = np.hstack((pc.a[N], pc.a[N]))
    # Multiply rows of pc.b with entries of Ds
    _ = pc.a[1:] + 2 * (pc.b[1:].T * Ds).T
    a_intp[:N, pc._nm:] = _

    b_intp = np.empty((pc.N+1, 2 * pc._nm))
    b_intp[:, 0:pc._nm] = pc.b
    b_intp[:N, pc._nm:] = pc.b[1:]
    b_intp[N] = np.hstack((pc.b[N], pc.b[N]))

    c_intp = np.empty((pc.N+1, 2 * pc._nm))
    c_intp[:, 0:pc._nm] = pc.c
    c_intp[:N, pc._nm:] = pc.c[1:]
    c_intp[N] = np.hstack((pc.c[N], pc.c[N]))

    # Equality part
    abar_intp = np.empty((pc.N+1, 2 * pc._neq))
    abar_intp[:, 0:pc._neq] = pc.abar
    abar_intp[N] = np.hstack((pc.abar[N], pc.abar[N]))

    bbar_intp = np.empty((pc.N+1, 2 * pc._neq))
    bbar_intp[:, 0:pc._neq] = pc.bbar
    bbar_intp[:N, pc._neq:] = pc.bbar[1:]
    bbar_intp[N] = np.hstack((pc.bbar[N], pc.bbar[N]))

    cbar_intp = np.empty((pc.N+1, 2 * pc._neq))
    cbar_intp[:, 0:pc._neq] = pc.cbar
    cbar_intp[:N, pc._neq:] = pc.cbar[1:]
    cbar_intp[N] = np.hstack((pc.cbar[N], pc.cbar[N]))

    D_intp = np.zeros((pc.N+1, 2 * pc._neq, 2 * pc._nv))
    D_intp[:, 0:pc._neq, 0:pc._nv] = pc.D
    D_intp[:N, pc._neq: 2 * pc._neq, pc._nv: 2 * pc._nv] = pc.D[1:]
    D_intp[N, 0:pc._neq, 0:pc._nv] = pc.D[N]
    D_intp[N, pc._neq: 2 * pc._neq, pc._nv: 2 * pc._nv] = pc.D[N]

    l_intp = np.empty((pc.N+1, 2 * pc._nv))
    l_intp[:, 0:pc._nv] = pc.l
    l_intp[:N, pc._nv:] = pc.l[1:]
    l_intp[N] = np.hstack((pc.l[N], pc.l[N]))

    h_intp = np.empty((pc.N+1, 2 * pc._nv))
    h_intp[:, 0:pc._nv] = pc.h
    h_intp[:N, pc._nv:] = pc.h[1:]
    h_intp[N] = np.hstack((pc.h[N], pc.h[N]))

    _ = pc.abar[1:] + 2 * (pc.bbar[1:].T * Ds).T
    abar_intp[:N, pc._neq:] = _

    # Inequality
    G_intp = np.empty((pc.N+1, 2 * pc.niq, 2 * pc._nv))
    G_intp[:, 0:pc.niq, 0:pc._nv] = pc.G
    G_intp[:N, pc.niq: 2 * pc.niq, pc._nv: 2 * pc._nv] = pc.G[1:]

    lG_intp = np.empty((pc.N+1, 2 * pc.niq))
    lG_intp[:, :pc.niq] = pc.lG
    lG_intp[:N, pc.niq:] = pc.lG[1:]
    hG_intp = np.empty((pc.N+1, 2 * pc.niq))
    hG_intp[:, :pc.niq] = pc.hG
    hG_intp[:N, pc.niq:] = pc.hG[1:]

    G_intp[N, 0:pc.niq, 0:pc._nv] = pc.G[N]
    G_intp[N, pc.niq: 2 * pc.niq, pc._nv: 2 * pc._nv] = pc.G[N]
    lG_intp[N] = np.hstack((pc.lG[N], pc.lG[N]))
    hG_intp[N] = np.hstack((pc.hG[N], pc.hG[N]))

    return PathConstraint(a=a_intp, b=b_intp, c=c_intp,
                          abar=abar_intp, bbar=bbar_intp, cbar=cbar_intp,
                          D=D_intp, l=l_intp, h=h_intp,
                          G=G_intp, lG=lG_intp, hG=hG_intp,
                          name=pc.name, ss=pc.ss)


def create_full_contact_path_constraint(path, ss, robot, stance):
    """ Rigid-body dynamics + Colomb frictional model.

    Args:
       path         : An Interpolator.
       ss           : A np.ndarray. Grid points.
       robot        : A Pymanoid.Humanoid. Used for dynamics computation.
                              Torque bound is taken from robot.rave.
       stance       : A Pymanoid.Stance. Used for wrench constraint.

    Returns:
       res: A PathConstraint.

    NOTE:
    The dynamics equation of a robot is given by:
                   M(q) qdd + qd' C(q) qd + g(q) = tau + sum(J_i(q, p_i) w_i),

    where
       q, qd, qdd: robot joint position, velocity and acceleration.
       tau: robot joint torque.
       w_i: the i-th local contact wrench acting on a link on the robot.
       p_i: the origin of the wrench w_i.
       J_i: the wrench Jacobian at p_i of the link w_i acts on.

    The slack variable is given by
                        v := [tau', w_1', w_2', ...]'.

    """
    N = len(ss) - 1
    q = path.eval(ss)
    qs = path.evald(ss)
    qss = path.evaldd(ss)
    torque_bnd = robot.rave.GetDOFTorqueLimits()
    dof = path.dof

    neq = dof
    nv = dof + 6 * len(stance.contacts)
    niq = sum(co.wrench_face.shape[0] for co in stance.contacts)

    abar = np.zeros((N+1, neq))
    bbar = np.zeros((N+1, neq))
    cbar = np.zeros((N+1, neq))
    D = np.zeros((N+1, neq, nv))
    l = np.zeros((N+1, nv))
    h = np.zeros((N+1, nv))
    G = np.zeros((N+1, niq, nv))
    lG = np.zeros((N+1, niq))
    hG = np.zeros((N+1, niq))

    for i in range(N+1):
        # t1,t2,t3,t4 are coefficients of the Path-Torque formulae
        t1, t3, t4 = Rave.inv_dyn(robot.rave, q[i], qs[i], qs[i])
        t2, _, _ = Rave.inv_dyn(robot.rave, q[i], qs[i], qss[i])
        abar[i] = t1
        bbar[i] = t2 + t3
        cbar[i] = t4
        D[i, :, :dof] = np.eye(dof)
        r = 0
        for con in stance.contacts:
            J_wrench = compute_jacobian_wrench(robot.rave, con.link, con.p)
            D[i, :, dof + r: dof + r + 6] = J_wrench.T
            r += 6
        l[i, :dof] = - torque_bnd
        h[i, :dof] = + torque_bnd
        l[i, dof:] = - INFTY  # Safety bounds.
        h[i, dof:] = INFTY

        _G_block = block_diag(*[co.wrench_face for co in stance.contacts])
        G[i] = np.hstack((np.zeros((niq, dof)), _G_block))
        lG[i, :] = - INFTY
        hG[i, :] = 0
    return PathConstraint(abar=abar, bbar=bbar, cbar=cbar, D=D,
                          l=l, h=h, lG=lG, G=G,
                          hG=hG, ss=ss, name='FullContactStability')


def create_pymanoid_contact_stability_path_constraint(
        path, ss, robot, contact_set, g):
    """Contact-stable constraint.

    Total rate of change of momentum of robot must be generated by the
    provided contacts with respect to the [0, 0, 0] fixed point.

    Args:
       path         : An Interpolator.
       robot        : A Pymanoid.Humanoid.
                Used for angular momentum computation.
       contact_set  : A Pymanoid.ContactSet.
                Used for wrench computation.
       ss           : A np.ndarray.
                Discretize position.

    Returns:
       res: A PathConstraint.

    """
    N = len(ss) - 1
    q = path.eval(ss)
    qs = path.evald(ss)
    qss = path.evaldd(ss)
    pO = np.zeros(3)  # fixed point

    F = contact_set.compute_wrench_face(pO)
    niq = F.shape[0]  # Number of inequalities
    m = robot.mass
    a = np.zeros((N + 1, niq))
    b = np.zeros((N + 1, niq))
    c = np.zeros((N + 1, niq))

    # Let O be a chosen pO, EL equation yields
    #     w^gi + w^c = 0,
    # where w^gi is the gravito-inertial wrench taken at O, w^c is the
    # contact wrench taken at O.
    for i in range(N+1):
        robot.set_dof_values(q[i])
        J_COM = robot.compute_com_jacobian()
        H_COM = robot.compute_com_hessian()
        J_L = robot.compute_angular_momentum_jacobian(pO)
        H_L = robot.compute_angular_momentum_hessian(pO)
        a_P = m * np.dot(J_COM, qs[i])
        b_P = m * (np.dot(J_COM, qss[i]) +
                   np.dot(qs[i], np.dot(H_COM, qs[i])))
        a_L = np.dot(J_L, qs[i])
        b_L = np.dot(J_L, qss[i]) + np.dot(qs[i], np.dot(H_L, qs[i]))
        pG = robot.com
        a[i] = np.dot(F, np.r_[a_P, a_L])
        b[i] = np.dot(F, np.r_[b_P, b_L])
        c[i] = - np.dot(F, np.r_[m * g, m * np.cross(pG, g)])

    return PathConstraint(a, b, c, name="ContactStability", ss=ss)


def create_rave_re_torque_path_constraint(path, ss, robot, J_lc,
                                          torque_bnd=None):
    """Torque bounds for robots under loop closure constraints.

    Loop closure constraint: Only virtual displacements
    dq satisfying

          J_lc(q) dq = 0,

    is admissible.

    Args:
       path    :  An Interpolator.
       ss      :  A np.ndarray. Shape (N+1, )
            Contrains discretized position.
       robot   :  An OpenRAVE.Robot.
       J_lc    : A mapping q -> an (d, dof) ndarray.

    Returns:
       pc  : A Path Constraint.
    """
    N = len(ss) - 1
    q = path.eval(ss)
    qs = path.evald(ss)
    qss = path.evaldd(ss)
    dof = path.dof

    if torque_bnd is None:
        torque_bnd = robot.GetDOFTorqueLimits()
    a = np.zeros((N + 1, dof))
    b = np.zeros((N + 1, dof))
    c = np.zeros((N + 1, dof))
    D = np.zeros((N + 1, dof, dof))
    l = -torque_bnd * np.ones((N + 1, dof))
    h = torque_bnd * np.ones((N + 1, dof))

    for i in range(N + 1):
        qi = q[i]
        qsi = qs[i]
        qssi = qss[i]
        # Column of N span the null space of J_lc(q)
        J_lp = J_lc(qi)
        u, s, v = np.linalg.svd(J_lp)
        # Collect column index
        s_full = np.zeros(dof)
        s_full[:s.shape[0]] = s
        # Form null matrix
        N = v[s_full < 1e-5].T
        D[i][:N.shape[1]] = N.T

        # t1,t2,t3,t4 are coefficients of the Path-Torque formulae
        t1, t3, t4 = Rave.inv_dyn(robot, qi, qsi, qsi)
        t2, _, _ = Rave.inv_dyn(robot, qi, qsi, qssi)

        a[i] = np.dot(D[i], t1)
        b[i] = np.dot(D[i], t2 + t3)
        c[i] = np.dot(D[i], t4)

    return PathConstraint(abar=a, bbar=b, cbar=c, D=D, l=l, h=h,
                          name="RedundantTorqueBounds", ss=ss)


def create_rave_torque_path_constraint(path, ss, robot):
    """Torque bounds for an OpenRAVE robot.

    Path-Torque constraint has the form

            sdd * M(q) qs(s)
          + sd ^ 2 [M(q) qss(s) + qs(s)^T C(q) qs(s)]
          + g(q(s)) = tau

            taumin <= tau <= taumax

    As canonical constraint.
            sdd * Ai + sd^2 Bi + Ci <= 0

    Args:
       path: An Interpolator.
       ss: A np.ndarray.
               Contains discretized positions.
       robot: An OpenRAVE robot.

    Returns:
       _: A PathConstraint.
    """
    N = len(ss) - 1
    q = path.eval(ss)
    qs = path.evald(ss)
    qss = path.evaldd(ss)
    dof = path.dof

    tau_bnd = robot.GetDOFTorqueLimits()
    a = np.zeros((N + 1, 2 * dof))
    b = np.zeros((N + 1, 2 * dof))
    c = np.zeros((N + 1, 2 * dof))
    for i in range(N + 1):
        qi = q[i]
        qsi = qs[i]
        qssi = qss[i]

        # t1,t2,t3,t4 are coefficients of the Path-Torque formulae
        t1, t3, t4 = Rave.inv_dyn(robot, qi, qsi, qsi)
        t2, _, _ = Rave.inv_dyn(robot, qi, qsi, qssi)

        a[i, :dof] = t1
        a[i, dof:] = -t1
        b[i, :dof] = t2 + t3
        b[i, dof:] = -t2 - t3
        c[i, :dof] = t4 - tau_bnd
        c[i, dof:] = -t4 - tau_bnd

    return PathConstraint(a, b, c, name="TorqueBounds", ss=ss)


def create_velocity_path_constraint(path, ss, vlim):
    """ Return joint velocities bound.

    Velocity constraint has the form:
                0 * ui +  1 * xi - sdmax^2 <= 0
                0      + -1      + sdmin^2 <= 0

    Args:
    -----
        path: Interpolator
        vlim: ndarray.
            Shaped (dof, 2) indicating velocity limits.
        ss: ndarray.
            Shaped (N+1,), the discretization knot points.

    Returns:
    --------
        pc: A `PathConstraint`.
    """
    qs = path.evald(ss)
    # Return resulti from cython version
    a, b, c = _create_velocity_constraint(qs, vlim)
    return PathConstraint(a, b, c, name="Velocity", ss=ss)


def create_acceleration_path_constraint(path, ss, alim):
    """ Joint accelerations bound.

    Acceleration constraint form:

                qs(si) ui + qss(si) sdi ^ 2 - qdmax <= 0
               -qs(si) ui - qss(si) sdi ^ 2 + qdmin <= 0

    Args:
    -----
        path      : An Interpolator.
        alim      : A ndarray. Shape (dof, 2). Acceleration limits.
        ss        : A ndarray. Shape (N+1,) Discretization knot points.

    Returns:
    --------
         _: A canonical PathConstraint object.
    """
    N = len(ss) - 1
    qs = path.evald(ss)
    qss = path.evaldd(ss)

    alim = np.array(alim)
    dof = path.dof  # dof

    if dof != 1:  # Non-scalar
        a = np.hstack((qs, -qs))
        b = np.hstack((qss, -qss))
        c = np.zeros((N + 1, 2 * dof))
        c[:, :dof] = -alim[:, 1]
        c[:, dof:] = alim[:, 0]
    else:
        a = np.vstack((qs, -qs)).T
        b = np.vstack((qss, -qss)).T
        c = np.zeros((N + 1, 2))
        c[:, 0] = -alim[:, 1]
        c[:, 1] = alim[:, 0]

    return PathConstraint(a, b, c, name="Acceleration", ss=ss)

###############################################################################
#                   PathParameterization Algorithms/Objects                   #
###############################################################################


def compute_trajectory_gridpoints(path, ss, us, xs):
    """ Convert grid points to a trajectory.

    Args:
    -----
    path: An Interpolator.
    ss: A (N+1,) ndarray. Array of gridpoints.
    us: A (N,) ndarray. Array of controls.
    xs: A (N+1,) ndarray. Array of squared velocities.


    Returns:
    --------
    ts: A (N+1, ) ndarray. Time at each gridpoints.
    q: A (N+1, dof) ndarray. Joint positions at each gridpoints.
    qd: A (N+1, dof) ndarray. Joint velocities at each gridpoints.
    qdd: A (N+1, dof) ndarray. Joint accelerations at each gridpoints.
    """
    ts = np.zeros_like(ss)
    N = ss.shape[0] - 1
    sd = np.sqrt(xs)
    sdd = np.hstack((us, us[-1]))
    for i in range(N):
        ts[i+1] = (sd[i+1] - sd[i]) / us[i] + ts[i]
    q = path.eval(ss)
    qs = path.evald(ss)  # derivative w.r.t [path position] s
    qss = path.evaldd(ss)
    array_mul = lambda v_arr, s_arr: np.array(
        [v_arr[i] * s_arr[i] for i in range(N+1)])
    qd = array_mul(qs, sd)
    qdd = array_mul(qs, sdd) + array_mul(qss, sd ** 2)
    return ts, q, qd, qdd


def compute_trajectory_points(path, sgrid, ugrid, xgrid, dt=1e-2):
    """ Compute trajectory with uniform time-spacing.

    Args:
    -----
    path: An Interpolator.
    sgrid: A (N+1,) ndarray. Array of gridpoints.
    ugrid: A (N,) ndarray. Array of controls.
    xgrid: A (N+1,) ndarray. Array of squared velocities.
    dt: A float. Spacing between subsequent time step.


    Returns:
    --------
    tsample: A (N+1, ) ndarray. Trajectory time stamps.
    q: A (N+1, dof) ndarray.
    qd: A (N+1, dof) ndarray.
    qdd: A (N+1, dof) ndarray.
    """
    tgrid = np.zeros_like(sgrid)
    N = sgrid.shape[0] - 1
    sdgrid = np.sqrt(xgrid)
    for i in range(N):
        tgrid[i+1] = (sdgrid[i+1] - sdgrid[i]) / ugrid[i] + tgrid[i]
    # sampled points on trajectory
    tsample = np.arange(tgrid[0], tgrid[-1], dt)
    ssample = np.zeros_like(tsample)
    xsample = np.zeros_like(tsample)
    sdsample = np.zeros_like(tsample)
    usample = np.zeros_like(tsample)
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
        res = np.zeros_like(vectors)
        for i in range(scalars.shape[0]):
            res[i] = vectors[i] * scalars[i]
        return res

    qd = array_mul(qs, sdsample)
    qdd = array_mul(qs, usample) + array_mul(qss, sdsample ** 2)
    return tsample, q, qd, qdd


class qpOASESPPSolver(object):
    """Implementation of TOPP-RA using qpOASES solver.

    This class translates a set of `PathConstraint` to the form needed
    by qpOASES form.

    Attributes:
    ----------
    I0: ndarray.
    IN: ndarray.
    ss: ndarray. Discretized path positions.
    nm: An Int. Dimension of the canonical constraint part.
    nv: An Int. Dimension of the combined slack.
    niq: An Int. Dimension of the inequalities on slack.
    neq: An Int. Dimension of the equalities on slack.
    nv: An Int. (qpOASES) Dimension of the optimization variable.
    nC: An Int. (qpOASES) Dimension of the constraint matrix.

    qpOASES matrices: coefficient matrices to be used with qpOASES solver.
    ----------------
            min     1/2 (u, x, v)^T H (u, x, v) + g^T (u, x, v)
            s.t.    lA <= A (u, x, v) <= hA
                    l  <=   (u, x, v) <= h
    A: ndarray.
    lA: ndarray.
    hA: ndarray.
    l: ndarray.
    h: ndarray.
    H: ndarray.
    g: ndarray.

    Matrices (lA, A, hA) is equivalent to all `PathConstraint(s)`'
    constraint.

    Note that the first `self.nop` row of A are all zeros. These are
    operational rows, used to specify additional constraints required
    for computing controllable/reachable sets.

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
        """ Initialization.

        Args:
        -----
        constraint_set:   A list of PathConstraint.
        """
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

        # Setup solvers
        self._init_qpoaes_solvers(verbose)

        summary_msg = """
Initialize Path Parameterization instance
------------------------------
\t N                  : {:8d}
\t No. of constraints : {:8d}
\t No. of slack var   : {:8d}
\t No. of equalities  : {:8d}
\t No. of inequalities: {:8d}
""".format(self.N, len(self.constraint_set),
           self.nv, self.neq, self.niq)
        if verbose:
            print summary_msg

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
        """ Starting interval at s_0
        """
        self.I0 = I0

    def set_goal_interval(self, IN):
        """ Desired goal interval at s_N
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
            options.printLevel = PrintLevel.HIGH
        else:
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

        Args:
        ----

        eps: A Float. The safety margin guarded againts numerical
                  error. This number needs not be too high.

        Returns:
        -------

        controllable: A Bool. True if K(0, IN) is not empty.
        """
        self._reset_operational_matrices()
        self.nWSR_up = np.ones((self.N+1, 1), dtype=int) * self.nWSR_cnst
        self.nWSR_down = np.ones((self.N+1, 1), dtype=int) * self.nWSR_cnst
        xmin, xmax = self.proj_x_admissible(self.N, self.IN[0],
                                            self.IN[1], init=True)
        if xmin is None:
            print "Unable to project the interval IN back to feasible set."
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
                print "Find controllable set K({:d}, IN failed".format(i)
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
            print "Unable to project the interval I0 back to feasibility"
            return False
        else:
            self._L[0, 1] = xmax
            self._L[0, 0] = xmin
        for i in range(self.N):
            init = (True if i <= 1 else False)
            xmin_nx, xmax_nx = self.reach(  # Next step, unprojected
                i, self._L[i, 0], self._L[i, 1], init=init)
            if xmin_nx is None:
                print "Forward propagation from L{:d} failed ".format(i)
                return False
            xmin_pr, xmax_pr = self.proj_x_admissible(  # Projected
                i + 1, xmin_nx, xmax_nx, init=init)
            if xmin_pr is None:
                print "Projection for L{:d} failed".format(i)
                return False
            else:
                self._L[i + 1, 1] = xmax_pr
                self._L[i + 1, 0] = xmin_pr
        return True

    def solve_topp(self, save_solutions=False, reg=0.):
        """Solve for the time-optimal path-parameterization

        Args:
        ----
        save_solutions: A Bool. Save solutions of each step.
        reg: A Float. Regularization gain.


        Returns:
        -------
        us: An ndarray. Contains the TOPP's controls.
        Xs: An ndarray. Contains the TOPP's squared velocities.

        """
        if save_solutions:
            self._xfulls = np.empty((self.N, self.nV))
            self._yfulls = np.empty((self.N, self.nC))
        # Backward pass
        controllable = self.solve_controllable_sets()  # Check controllability
        # Check for solvability
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
            print """
Computing one-step failed.

    INFO:
    ----
        i                     = {}
        xmin                  = {}
        xmax                  = {}
        warm_start            = {}
        upper LP solve status = {}
        lower LP solve status = {}
""".format(i, xmin, xmax, init, res_up, res_down)

            return None, None

        # extract solution
        self.solver_up.getPrimalSolution(self._xfull)
        xmax_i = self._xfull[1]
        self.solver_down.getPrimalSolution(self._xfull)
        xmin_i = self._xfull[1]
        return xmin_i, xmax_i

    def reach(self, i, xmin, xmax, init=False):
        """Compute the reach set from [xmin, xmax] at stage i.

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
            print """
Computing reach set failed.

    INFO:
    ----
        i                     = {}
        xmin                  = {}
        xmax                  = {}
        warm_start            = {}
        upper LP solve status = {}
        lower LP solve status = {}
""".format(i, xmin, xmax, init, res_up, res_down)
            return None, None

        # extract solution
        xmax_i = -self.solver_up.getObjVal()
        xmin_i = self.solver_down.getObjVal()
        return xmin_i, xmax_i

    def proj_x_admissible(self, i, xmin, xmax, init=False):
        """Project [xmin, xmax] back to Omega_i.

        If the projection is not feasible (for example when xmin >
        xmax), then return None, None.
        """

        self.A[i, 0, 1] = 1
        self.A[i, 0, 0] = 0.
        self.lA[i, 0] = xmin
        self.hA[i, 0] = xmax

        # upper bound
        nWSR_up = np.array([self.nWSR_cnst])
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
            print """
Computing projection failed.

    INFO:
    ----
        i                     = {}
        xmin                  = {}
        xmax                  = {}
        warm_start            = {}
        upper LP solve status = {}
        lower LP solve status = {}
""".format(i, xmin, xmax, init, res_up, res_down)
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
            print "Non-optimal solution at i={0}. Returning default.".format(i)
            return None, None

        # extract solution
        self.solver_up.getPrimalSolution(self._xfull)
        # self.solver_up.getDualSolution(self._yfull)  # cause failure
        u_topp = self._xfull[0]
        return u_topp, x + 2 * self.Ds[i] * u_topp
