"""This module constraints the PathConstraint object and factory
functions to generate them.

"""
import numpy as np
from enum import Enum
from utils import inv_dyn, compute_jacobian_wrench
from _CythonUtils import _create_velocity_constraint
from scipy.linalg import block_diag
from TOPP import INFTY
import logging
logger = logging.getLogger(__name__)



class PathConstraintKind(Enum):
    Canonical = 0
    TypeI = 1
    TypeII = 2


class PathConstraint(object):
    """Discretized constraint on a path.

    Parameters
    ----------
    a     : (N+1, neq)     ndarray or None, optional
    b     : (N+1, neq)     ndarray or None, optional
    c     : (N+1, neq)     ndarray or None, optional
    abar  : (N+1, neq)     ndarray or None, optional
    bbar  : (N+1, neq)     ndarray or None, optional
    cbar  : (N+1, neq)     ndarray or None, optional
    lG    : (N+1, niq)     ndarray or None, optional
    hG    : (N+1, niq)     ndarray or None, optional
    G     : (N+1, niq, nv) ndarray or None, optional
    l     : (N+1, nv)      ndarray or None, optional
    h     : (N+1, nv)      ndarray or None, optional
    name : String.
           Represents the name of the constraint.

    Attributes:
    -----------
    _nm : Int.
          Dimension of non-redundant inequalities.
    _nv : Int.
          Dimension of slack variable.
    _neq : Int.
           Dimension of equality constraint.
    _niq : Int.
           Dimension of inequality constraint.
    N : Int.
        Number of discretization segments.

    Notes
    -----

    In the most general setting, a PathConstraint object describes the
    following constraints:

                a(s) u    + b(s) x    + c(s)    <= 0
                abar(s) u + bbar(s) x + cbar(s)  = D(s) v

                lG(s) <=     G(s) v             <= hG(s)
                l(s)  <=          v             <= h(s)

    where u is the path acceleration, x is the squared path velocity
    and v is a slack variable whose physical meanings depend on the
    nature of the constraint.

    In practice, a PathConstraint object takes one in three following
    states

    1. *Canonical*: Only matrices a, b, c are not None. All other
       matrices are None.
    2. *Non-Canonical Type I*: Matrices abar, bbar, cbar, D, l, h are
       not None. All other matrices are None.
    3. *Non-Canonical Type II*: Matrices abar, bbar, cbar, D, l, h,
       lG, G, hG are not None. All other matrices, which are a, b, c
       are None.

    The PathConstraint object can represent both the collocated
    constraints and the interpolated constraints. Note that the
    factory functions `create_[..]_path_constraint` only return the
    collocated version. To create a interpolated PathConstraint, one
    needs to use the function `interpolate_constraint`.

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
        t1, t3, t4 = inv_dyn(robot.rave, q[i], qs[i], qs[i])
        t2, _, _ = inv_dyn(robot.rave, q[i], qs[i], qss[i])
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
        t1, t3, t4 = inv_dyn(robot, qi, qsi, qsi)
        t2, _, _ = inv_dyn(robot, qi, qsi, qssi)

        a[i] = np.dot(D[i], t1)
        b[i] = np.dot(D[i], t2 + t3)
        c[i] = np.dot(D[i], t4)

    return PathConstraint(abar=a, bbar=b, cbar=c, D=D, l=l, h=h,
                          name="RedundantTorqueBounds", ss=ss)


def create_rave_torque_path_constraint(path, ss, robot):
    """Torque bounds for an OpenRAVE robot.

    Notes
    -----
    Path-Torque constraint has the form

            sdd * M(q) qs(s)
          + sd ^ 2 [M(q) qss(s) + qs(s)^T C(q) qs(s)]
          + g(q(s)) = tau

            taumin <= tau <= taumax

    As canonical constraint.
            sdd * Ai + sd^2 Bi + Ci <= 0

    Parameters
    ----------
    path: toppra's Interpolator
          Represents the underlying geometric path.
    ss: ndarray
        Discretization gridpoints.
    robot: OpenRAVE robot
           the robot model to obtain dynamics matrices


    Returns
    -------
    out: PathConstraint
         The equivalnet path constraint
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
        t1, t3, t4 = inv_dyn(robot, qi, qsi, qsi)
        t2, _, _ = inv_dyn(robot, qi, qsi, qssi)

        a[i, :dof] = t1
        a[i, dof:] = -t1
        b[i, :dof] = t2 + t3
        b[i, dof:] = -t2 - t3
        c[i, :dof] = t4 - tau_bnd
        c[i, dof:] = -t4 - tau_bnd

    logger.info("Torque bounds for OpenRAVE robot generated.")
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

