##########################################################################
# This file contains test suites for `PathConstraint` factory functions. #
##########################################################################
try:
    from os.path import expanduser
    import sys
    sys.path.insert(0, expanduser('~/git/pymanoid'))
    import pymanoid
    skip_pymanoid_tests = True  # Alway skip this test
except ImportError:
    skip_pymanoid_tests = True


import toppra as fa
import numpy as np
import numpy.testing as npt
import openravepy as orpy
from toppra import (PolyPath, SplineInterpolator,
                    normalize, PathConstraintKind, TINY, SMALL)
import pytest
import cvxpy as cvx
# some constants
from testingUtils import (U_LOWER, U_HIGHER, X_LOWER, X_HIGHER)

VERBOSE = False
DEBUG = False


def set_OpenRAVE_debug_warn():
    orpy.RaveSetDebugLevel(orpy.DebugLevel.Fatal)
    orpy.misc.InitOpenRAVELogging()


@pytest.fixture(scope="class", params=[2, 6], name='velocity_pc_data')
def create_velocity_pc_fixtures(request):
    """Parameterized fixture to test Velocity constraint.

    Return:
    -------
      data: A tuple. Contains path, ss, vim.
      pc: A `PathConstraint`.

    """
    if request.param == 2:
        coeff = [[1., 2, 3], [-2., -3., 4., 5.]]
        pi = PolyPath(coeff)
        ss = np.linspace(0, 0.75, 4)
        vlim = np.array([[-1., 2], [-2., 2]])
        pc_vel = fa.create_velocity_path_constraint(pi, ss, vlim)
        data = (pi, ss, vlim)
        return data, pc_vel

    if request.param == 6:
        np.random.seed(10)
        N = 100
        way_pts = np.random.randn(10, 6)
        pi = SplineInterpolator(np.linspace(0, 1, 10), way_pts)
        ss = np.linspace(0, 1, N + 1)
        vlim_ = np.random.rand(6) * 10 + 2.
        vlim = np.vstack((-vlim_, vlim_)).T
        pc_vel = fa.create_velocity_path_constraint(pi, ss, vlim)
        data = (pi, ss, vlim)
        return data, pc_vel


class TestFunc_create_velocity_path_constraint(object):
    """Test suite for function `create_velocity_path_constraint`.

    Tests:
    ------

    1. syntactic: the object return should have correct dimension.

    2. constraint satisfaction: the `PathConstraint` returned should
    be consistent with the data.

    """
    def test_syntax(self, velocity_pc_data):
        """ Syntactic correct.
        """
        data, pc = velocity_pc_data
        path, ss, vlim = data
        assert np.allclose(ss, pc.ss)

        # Canonical cnst assertions
        assert np.allclose(pc.abar, np.empty((pc.N + 1, 0)))
        assert np.allclose(pc.bbar, np.empty((pc.N + 1, 0)))
        assert np.allclose(pc.cbar, np.empty((pc.N + 1, 0)))
        assert np.allclose(pc.D, np.empty((pc.N + 1, 0, 0)))
        assert np.allclose(pc.lG, np.empty((pc.N + 1, 0)))
        assert np.allclose(pc.hG, np.empty((pc.N + 1, 0)))
        assert np.allclose(pc.G, np.empty((pc.N + 1, 0, 0)))
        assert np.allclose(pc.l, np.empty((pc.N + 1, 0)))
        assert np.allclose(pc.h, np.empty((pc.N + 1, 0)))

        # dimension
        assert pc.nm == 2  # canonical part
        assert pc.neq == 0  # non-canonical part
        assert pc.niq == 0
        assert pc.nv == 0

        assert pc.a.shape == (pc.N + 1, 2)
        assert pc.b.shape == (pc.N + 1, 2)
        assert pc.c.shape == (pc.N + 1, 2)
        assert pc.kind == PathConstraintKind.Canonical

    def test_constraint_satisfaction(self, velocity_pc_data):
        """ Test constraint satisfaction with cvxpy.

        3. They should agree.
        """
        data, pc = velocity_pc_data
        path, ss, vlim = data
        assert np.allclose(ss, pc.ss)

        qs = path.evald(ss)

        u = cvx.Variable(1)
        x = cvx.Variable(1)
        sd = cvx.Variable()

        for i in range(0, pc.N + 1):
            # 1. Compute max sd from the constraint
            constraints = [u * pc.a[i] + x * pc.b[i] + pc.c[i] <= 0]
            obj = cvx.Maximize(x)
            prob = cvx.Problem(obj, constraints)
            prob.solve(solver=cvx.ECOS, abstol=1e-9)
            res_pcmax = np.array([u.value, x.value])
            prob = cvx.Problem(cvx.Minimize(x), constraints)
            prob.solve(solver=cvx.ECOS, abstol=1e-9)
            res_pcmin = np.array([u.value, x.value])

            # 2. Compute max sd from the data
            constraints = [qs[i] * sd <= vlim[:, 1],
                           qs[i] * sd >= vlim[:, 0],
                           sd >= 0]
            prob = cvx.Problem(cvx.Maximize(sd), constraints)
            prob.solve(solver=cvx.ECOS, abstol=1e-9)
            res_datamax = np.array([u.value, sd.value ** 2])
            prob = cvx.Problem(cvx.Minimize(sd), constraints)
            prob.solve(solver=cvx.ECOS, abstol=1e-9)
            res_datamin = np.array([u.value, sd.value ** 2])
            # 3. They should agree
            assert np.allclose(res_datamax, res_pcmax)
            assert np.allclose(res_datamin, res_pcmin)


@pytest.fixture(scope="class", params=[1, 2, 6], name='acceleration_pc_data')
def create_acceleration_pc_fixtures(request):
    """ Parameterized Acceleration path constraint.

    Return:
    -------
      data: A tuple. Contains path, ss, alim.
      pc: A `PathConstraint`.
    """
    if request.param == 1:  # Scalar
        pi = PolyPath([1, 2, 3])  # 1 + 2s + 3s^2
        ss = np.linspace(0, 1, 3)
        alim = (np.r_[-1., 1]).reshape(1, 2)  # Scalar case
        pc_vel = fa.create_acceleration_path_constraint(pi, ss, alim)
        data = (pi, ss, alim)
        return data, pc_vel

    if request.param == 2:
        coeff = [[1., 2, 3], [-2., -3., 4., 5.]]
        pi = PolyPath(coeff)
        ss = np.linspace(0, 0.75, 4)
        alim = np.array([[-1., 2], [-2., 2]])
        pc_vel = fa.create_acceleration_path_constraint(pi, ss, alim)
        data = (pi, ss, alim)
        return data, pc_vel

    if request.param == 6:
        np.random.seed(10)
        N = 20
        way_pts = np.random.randn(10, 6)
        pi = SplineInterpolator(np.linspace(0, 1, 10), way_pts)
        ss = np.linspace(0, 1, N + 1)
        vlim_ = np.random.rand(6)
        alim = np.vstack((-vlim_, vlim_)).T
        pc_vel = fa.create_acceleration_path_constraint(pi, ss, alim)
        data = (pi, ss, alim)
        return data, pc_vel


class TestFunc_create_acceleration_path_constraint(object):
    """Test suite for function `create_acceleration_path_constraint`.

    Tests:
    ------

    1. syntactic: the object return should have correct dimension.

    2. constraint satisfaction: the `PathConstraint` returned should
    be consistent with the data.

    """
    def test_syntax(self, acceleration_pc_data):
        """ Syntactic correct.
        """
        data, pc = acceleration_pc_data
        path, ss, vlim = data
        assert np.allclose(ss, pc.ss)

        # Canonical cnst assertions
        assert np.allclose(pc.abar, np.empty((pc.N + 1, 0)))
        assert np.allclose(pc.bbar, np.empty((pc.N + 1, 0)))
        assert np.allclose(pc.cbar, np.empty((pc.N + 1, 0)))
        assert np.allclose(pc.D, np.empty((pc.N + 1, 0, 0)))
        assert np.allclose(pc.lG, np.empty((pc.N + 1, 0)))
        assert np.allclose(pc.hG, np.empty((pc.N + 1, 0)))
        assert np.allclose(pc.G, np.empty((pc.N + 1, 0, 0)))
        assert np.allclose(pc.l, np.empty((pc.N + 1, 0)))
        assert np.allclose(pc.h, np.empty((pc.N + 1, 0)))

        # dimension
        assert pc.nm == path.dof * 2  # canonical part
        assert pc.neq == 0  # non-canonical part
        assert pc.niq == 0
        assert pc.nv == 0

        assert pc.a.shape == (pc.N + 1, path.dof * 2)
        assert pc.b.shape == (pc.N + 1, path.dof * 2)
        assert pc.c.shape == (pc.N + 1, path.dof * 2)
        assert pc.kind == PathConstraintKind.Canonical

    def test_constraint_satisfaction(self, acceleration_pc_data):
        """ Test constraint satisfaction with cvxpy.
        """
        data, pc = acceleration_pc_data
        path, ss, alim = data
        assert np.allclose(ss, pc.ss)

        qs = path.evald(ss)
        qss = path.evaldd(ss)

        u = cvx.Variable()
        x = cvx.Variable()

        for i in range(0, pc.N + 1):
            # Path Constraint
            constraints = [u * pc.a[i] + x * pc.b[i] + pc.c[i] <= 0,
                           x >= 0,
                           x <= 1000]  # to avoid inf

            res_pc = np.zeros((10, 2))
            for j in range(10):  # try 10 direction
                obj = cvx.Maximize(np.sin(float(j) / np.pi / 2) * u +
                                   np.sin(float(j) / np.pi / 2) * x)
                prob = cvx.Problem(obj, constraints)
                prob.solve(solver=cvx.CVXOPT)
                res_pc[j, :] = np.array([u.value, x.value])

            # Data
            res_data = np.zeros((10, 2))
            constraints = [qs[i] * u + qss[i] * x <= alim[:, 1],
                           qs[i] * u + qss[i] * x >= alim[:, 0],
                           x >= 0, x <= 1000]
            for j in range(10):  # try 10 direction
                obj = cvx.Maximize(np.sin(float(j) / np.pi / 2) * u +
                                   np.sin(float(j) / np.pi / 2) * x)
                prob = cvx.Problem(obj, constraints)
                prob.solve(solver=cvx.CVXOPT)
                res_data[j, :] = np.array([u.value, x.value])
            assert np.allclose(res_data, res_pc)


@pytest.fixture(scope="class", params=['fixed', 1, 2, 6],
                name='torque_pc_data')
def create_rave_torque_pc_fixtures(request):
    """Parameterized Rave Torque path constraint.

    Returns:
    --------
        data: A tuple. Contains path, ss, robot.
        pc: A `PathConstraint`.

    Params:
    -------

        If `param` is an int, generate a random trajectory using `param`
        as the random seed.

        If `param` is "fixed", generate a trajectory with constant joint
        values.

    """
    env = orpy.Environment()
    set_OpenRAVE_debug_warn()
    env.Load('robots/pumaarm.zae')
    # env.SetViewer('qtosg')
    robot = env.GetRobots()[0]
    dof = robot.GetDOF()
    robot.SetDOFTorqueLimits(np.ones(dof) * 100)

    if type(request.param) is int:
        np.random.seed(request.param)
        way_pts = np.random.randn(7, dof) * 0.5
        pi = SplineInterpolator(np.linspace(0, 1, 7), way_pts)
        N = 100
        ss = np.linspace(0, 2, N + 1)

        pc = fa.create_rave_torque_path_constraint(pi, ss, robot)
        yield (pi, ss, robot), pc
        orpy.RaveDestroy()

    elif request.param == 'fixed':
        np.random.seed(1)
        q_fixed = np.random.randn(dof)
        way_pts = [q_fixed for i in range(7)]
        pi = SplineInterpolator(np.linspace(0, 1, 7), way_pts)
        N = 100
        ss = np.linspace(0, 2, N + 1)
        pc = fa.create_rave_torque_path_constraint(pi, ss, robot)
        yield (pi, ss, robot), pc
        orpy.RaveDestroy()


class TestFunc_create_rave_torque_path_constraint(object):
    """Test suite for function `create_rave_torque_path_constraint`.

    Tests:
    ------

    1. syntactic: the object return should have correct dimension.

    2. constraint satisfaction: the `PathConstraint` returned should
    be consistent with the data.

    """
    def test_syntax(self, torque_pc_data):
        """ Syntactic correct.
        """
        data, pc = torque_pc_data
        path, ss, robot = data
        assert np.allclose(ss, pc.ss)

        # Canonical cnst assertions
        assert np.allclose(pc.abar, np.empty((pc.N + 1, 0)))
        assert np.allclose(pc.bbar, np.empty((pc.N + 1, 0)))
        assert np.allclose(pc.cbar, np.empty((pc.N + 1, 0)))
        assert np.allclose(pc.D, np.empty((pc.N + 1, 0, 0)))
        assert np.allclose(pc.lG, np.empty((pc.N + 1, 0)))
        assert np.allclose(pc.hG, np.empty((pc.N + 1, 0)))
        assert np.allclose(pc.G, np.empty((pc.N + 1, 0, 0)))
        assert np.allclose(pc.l, np.empty((pc.N + 1, 0)))
        assert np.allclose(pc.h, np.empty((pc.N + 1, 0)))

        # dimension
        assert pc.nm == path.dof * 2  # canonical part
        assert pc.neq == 0  # non-canonical part
        assert pc.niq == 0
        assert pc.nv == 0

        assert pc.a.shape == (pc.N + 1, path.dof * 2)
        assert pc.b.shape == (pc.N + 1, path.dof * 2)
        assert pc.c.shape == (pc.N + 1, path.dof * 2)
        assert pc.kind == PathConstraintKind.Canonical

    def test_constraint_satisfaction(self, torque_pc_data):
        """ Correctness assertion.
        """
        data, pc = torque_pc_data
        pi, ss, robot = data
        tau = robot.GetDOFTorqueLimits()

        # Correctness
        q = pi.eval(ss)
        qs = pi.evald(ss)
        qss = pi.evaldd(ss)

        u = cvx.Variable()
        x = cvx.Variable()

        # Remove vel/accel limits during inversedynamic
        robot.SetDOFAccelerationLimits(np.ones(robot.GetDOF()) * 100)
        robot.SetDOFVelocityLimits(np.ones(robot.GetDOF()) * 100)
        i = 0
        # From Data
        with robot:
            robot.SetDOFValues(q[i])
            robot.SetDOFVelocities(qs[i])
            tm1, tc1, tg1 = robot.ComputeInverseDynamics(
                qs[i], externalforcetorque=None, returncomponents=True)
            tm2, tc2, tg2 = robot.ComputeInverseDynamics(
                qss[i], externalforcetorque=None, returncomponents=True)
        constraints = [-tau <= tm1 * u + (tm2 + tc2) * x + tg1,
                       tm1 * u + (tm2 + tc2) * x + tg1 <= tau,
                       U_LOWER <= u, u <= U_HIGHER,
                       X_LOWER <= x, x <= X_HIGHER]

        res_data = np.zeros((10, 2))
        for j in range(10):  # try 10 direction
            obj = cvx.Maximize(np.sin(float(j) / np.pi / 2) * u +
                               np.sin(float(j) / np.pi / 2) * x)
            prob = cvx.Problem(obj, constraints)
            prob.solve(solver=cvx.CVXOPT)
            res_data[j, :] = np.array([u.value, x.value])

        # From PC
        constraints = [pc.a[i] * u + pc.b[i] * x + pc.c[i] <= 0,
                       U_LOWER <= u, u <= U_HIGHER,
                       X_LOWER <= x, x <= X_HIGHER]

        res_pc = np.zeros((10, 2))
        for j in range(10):  # try 10 direction
            obj = cvx.Maximize(np.sin(float(j) / np.pi / 2) * u +
                               np.sin(float(j) / np.pi / 2) * x)
            prob = cvx.Problem(obj, constraints)
            prob.solve(solver=cvx.CVXOPT)
            res_pc[j, :] = np.array([u.value, x.value])

        assert np.allclose(res_data, res_pc)


@pytest.fixture(name='rave_re_torque_data')
def create_rave_re_torque_fixtures():
    """
    """
    env = orpy.Environment()
    set_OpenRAVE_debug_warn()
    env.Load('robots/barrettwam.robot.xml')
    # env.SetViewer('qtosg')
    robot = env.GetRobots()[0]
    dof = robot.GetDOF()
    robot.SetDOFTorqueLimits(np.ones(dof) * 100)

    np.random.seed(1)
    way_pts = np.random.randn(5, dof) * 0.5
    pi = SplineInterpolator(np.linspace(0, 1, 5), way_pts)
    N = 20
    ss = np.linspace(0, 1, N + 1)

    # loop closure Jacobian
    def J_lp(q):
        with robot:
            robot.SetDOFValues(q)
            l_last = robot.GetLinks()[-1]
            return robot.ComputeJacobianTranslation(
                l_last.GetIndex(), l_last.GetGlobalCOM())

    # Loop closure
    pc_torque = fa.create_rave_re_torque_path_constraint(
        pi, ss, robot, J_lp)
    yield (pi, ss, robot, J_lp), pc_torque
    orpy.RaveDestroy()


class TestFunc_create_rave_re_torque_path_constraint(object):
    """Test suite for function `create_rave_re_torque_path_constraint`.

    Tests:
    ------

    1. syntactic: the object return should have correct dimension.

    2. constraint satisfaction: the `PathConstraint` returned should
    be consistent with the data.

    """
    def test_syntax(self, rave_re_torque_data):
        """ Syntactic correct.
        """
        data, pc = rave_re_torque_data
        path, ss, robot, J_lc = data
        assert np.allclose(ss, pc.ss)

        assert np.allclose(pc.a, np.empty((pc.N + 1, 0)))
        assert np.allclose(pc.b, np.empty((pc.N + 1, 0)))
        assert np.allclose(pc.c, np.empty((pc.N + 1, 0)))
        assert np.allclose(pc.lG, np.empty((pc.N + 1, 0)))
        assert np.allclose(pc.hG, np.empty((pc.N + 1, 0)))
        assert np.allclose(pc.G, np.empty((pc.N + 1, 0, pc.nv)))

        # dimension
        assert pc.nm == 0  # canonical part
        assert pc.neq == path.dof  # non-canonical part
        assert pc.niq == 0
        assert pc.nv == path.dof

        assert pc.abar.shape == (pc.N + 1, path.dof)
        assert pc.bbar.shape == (pc.N + 1, path.dof)
        assert pc.cbar.shape == (pc.N + 1, path.dof)
        assert pc.kind == PathConstraintKind.TypeI

    def test_slack_bounds(self, rave_re_torque_data):
        """ Test pc.l, pc.h vectors
        """
        data, pc = rave_re_torque_data
        path, ss, robot, J_lc = data
        tau_bnd = robot.GetDOFTorqueLimits()

        for i in range(pc.N + 1):
            npt.assert_allclose(pc.l[i], -tau_bnd)
            npt.assert_allclose(pc.h[i], tau_bnd)

    def test_null_space(self, rave_re_torque_data):
        """pc.D[i] must equals N.T, where N is orthogonal
        to the loop closure Jacobian.
        """
        data, pc = rave_re_torque_data
        pi, ss, robot, J_lc = data

        # Assert nullspace matrix at s index = 0
        q = pi.eval(ss)
        for i in range(pc.N + 1):
            Nmat = pc.D[i].T
            qi = q[i]
            nr, nc = Nmat.shape
            for i in range(nc):
                _ = np.dot(J_lc(qi), Nmat[:, i])
                npt.assert_allclose(np.linalg.norm(_), 0, atol=TINY)


@pytest.fixture(scope="class")
def pymanoid_fixture():
    """ Swaying.

    Scenario: Easy Sway for jvrc1 humanoid
    """
    import numpy as np

    import pymanoid
    from pymanoid import PointMass, Stance
    from pymanoid.contact import Contact
    from pymanoid.drawers import COMAccelConeDrawer
    from pymanoid.drawers import SEPDrawer
    from pymanoid.drawers import StaticWrenchDrawer
    from pymanoid.drawers import ZMPSupportAreaDrawer

    com_height = 0.9  # [m]
    z_polygon = 2.

    class COMSync(pymanoid.Process):

        def on_tick(self, sim):
            com_above.set_x(com_target.x)
            com_above.set_y(com_target.y)

    sim = pymanoid.Simulation(dt=0.03)
    if not VERBOSE:
        set_OpenRAVE_debug_warn()
    robot = pymanoid.robots.JVRC1('JVRC-1.dae', download_if_needed=True)
    if DEBUG:
        sim.set_viewer()
        sim.viewer.SetCamera([
            [0.60587192, -0.36596244, 0.70639274, -2.4904027],
            [-0.79126787, -0.36933163, 0.48732874, -1.6965636],
            [0.08254916, -0.85420468, -0.51334199, 2.79584694],
            [0., 0., 0., 1.]])

    robot.set_transparency(0.25)
    robot.set_dof_values([
        3.53863816e-02, 2.57657518e-02, 7.75586039e-02,
        6.35909636e-01, 7.38580762e-02, -5.34226902e-01,
        -7.91656626e-01, 1.64846093e-01, -2.13252247e-01,
        1.12500819e+00, -1.91496369e-01, -2.06646315e-01,
        1.39579597e-01, -1.33333598e-01, -8.72664626e-01,
        0.00000000e+00, -9.81307787e-15, 0.00000000e+00,
        -8.66484961e-02, -1.78097540e-01, -1.68940240e-03,
        -5.31698601e-01, -1.00166891e-04, -6.74394930e-04,
        -1.01552628e-04, -5.71121132e-15, -4.18037117e-15,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, -7.06534763e-01, 1.67723830e-01,
        2.40289101e-01, -1.11674923e+00, 6.23384177e-01,
        -8.45611535e-01, 1.39994759e-02, 1.17756934e-16,
        3.14018492e-16, -3.17943723e-15, -6.28036983e-16,
        -3.17943723e-15, -6.28036983e-16, -6.88979202e-02,
        -4.90099381e-02, 8.17415141e-01, -8.71841480e-02,
        -1.36966665e-01, -4.26226421e-02])

    com_target = PointMass(
        pos=[0., 0., com_height], mass=robot.mass, color='b', visible=False)
    com_above = pymanoid.Cube(0.02, [0.05, 0.04, z_polygon], color='b')
    # Attach links to the contacts.
    left_ankle = robot.rave.GetLink('L_ANKLE_P_S')
    right_ankle = robot.rave.GetLink('R_ANKLE_P_S')

    stance = Stance(
        com=com_target,
        left_foot=Contact(
            shape=robot.sole_shape,
            pos=[0.20, 0.15, 0.1],
            rpy=[0.4, 0, 0],
            friction=0.5,
            visible=True,
            link=left_ankle),
        right_foot=Contact(
            shape=robot.sole_shape,
            pos=[-0.2, -0.195, 0.],
            rpy=[-0.4, 0, 0],
            friction=0.5,
            visible=True,
            link=right_ankle))
    stance.bind(robot)
    robot.ik.solve()

    com_sync = COMSync()
    cone_drawer = COMAccelConeDrawer(stance, scale=0.05)
    sep_drawer = SEPDrawer(stance, z_polygon)
    wrench_drawer = StaticWrenchDrawer(com_target, stance)
    zmp_area_drawer = ZMPSupportAreaDrawer(stance, z_polygon)

    sim.schedule(robot.ik)
    sim.schedule_extra(com_sync)

    if DEBUG:
        sim.schedule_extra(cone_drawer)
        sim.schedule_extra(sep_drawer)
        sim.schedule_extra(wrench_drawer)
        sim.schedule_extra(zmp_area_drawer)

    # Generate a simple full-body path by tracking the COM path.
    try:
        ts, qs, _ = np.load("_temp_test_{}.npy".format(__name__))
    except:
        com_sync = COMSync()
        sim.schedule(robot.ik)
        sim.schedule_extra(com_sync)
        com_target.set_x(0)
        com_target.set_y(0)
        robot.ik.solve()
        ts = np.arange(0, 7, sim.dt)
        ps = np.array(
            [com_target.p + np.r_[0.1, 0.1, 0] * t for t in np.sin(ts)])
        qs = []
        for p in ps:
            qs.append(robot.q)
            com_target.set_pos(p)
            sim.step(1)
        sim.stop()
        qs = np.array(qs)
        np.save("_temp_test_{}.npy".format(__name__), [ts, qs, None])
        print """
Does not found stored path,
Generated and saved at {}""".format("_temp_test_{}.npy".format(__name__))

    path = SplineInterpolator(normalize(ts), qs)
    # Return
    N = 20
    ss = np.linspace(0, 1, N + 1)
    pc_contact = fa.create_pymanoid_contact_stability_path_constraint(
        path, ss, robot, stance, sim.gravity)
    yield (path, ss, robot, stance, sim.gravity), pc_contact
    orpy.RaveDestroy()
    pymanoid.sim.env = None
    print "\n ----> Destroy pymanoid's fixture"


@pytest.mark.skipif(skip_pymanoid_tests, reason="`Pymanoid` library not found!")
class Test_ContactStability(object):
    """Test suite for :func:`create_pymanoid_contact_stability_path_constraint`

    This suite will be skipped if the library `pymanoid` is not found.
    """
    def test_syntax(self, pymanoid_fixture):
        """ Check for syntactic errors.

        Parameters
        ----------
        pymanoid_fixture : A Fixture. Data for this test case.
        """
        data, pc_contact = pymanoid_fixture
        path, ss, humanoid, stance, g = data

        F = stance.compute_wrench_face(np.zeros(3))
        m = F.shape[0]

        # Canonical cnst assertions
        assert np.allclose(pc_contact.abar, np.empty((pc_contact.N + 1, 0)))
        assert np.allclose(pc_contact.bbar, np.empty((pc_contact.N + 1, 0)))
        assert np.allclose(pc_contact.cbar, np.empty((pc_contact.N + 1, 0)))
        assert np.allclose(pc_contact.D, np.empty((pc_contact.N + 1, 0, 0)))
        assert np.allclose(pc_contact.lG, np.empty((pc_contact.N + 1, 0)))
        assert np.allclose(pc_contact.hG, np.empty((pc_contact.N + 1, 0)))
        assert np.allclose(pc_contact.G, np.empty((pc_contact.N + 1, 0, 0)))
        assert np.allclose(pc_contact.l, np.empty((pc_contact.N + 1, 0)))
        assert np.allclose(pc_contact.h, np.empty((pc_contact.N + 1, 0)))

        assert pc_contact.a.shape == (pc_contact.N + 1, m)
        assert pc_contact.b.shape == (pc_contact.N + 1, m)
        assert pc_contact.c.shape == (pc_contact.N + 1, m)

        assert pc_contact.kind == PathConstraintKind.Canonical

        assert pc_contact.nm == m
        assert pc_contact.niq == 0
        assert pc_contact.neq == 0
        assert pc_contact.nv == 0

    def test_feasible_wrenches(self, pymanoid_fixture):
        """Verify consistency with `pymanoid`.

        The pair `(u, x)` satisfyng `pc_contact` must also return a
        feasible wrench from pymanoid.

        Parameters
        ----------
        pymanoid_fixture : A Fixture. Data for this test case.
        """
        data, pc_contact = pymanoid_fixture
        path, ss, humanoid, stance, g = data

        F = stance.compute_wrench_face(np.zeros(3))

        # Correctness
        q = path.eval(ss)
        qs = path.evald(ss)
        qss = path.evaldd(ss)

        u = cvx.Variable()
        x = cvx.Variable()
        for i in range(pc_contact.N + 1):
            constraints = [(pc_contact.a[i] * u + pc_contact.b[i] * x + pc_contact.c[i] <= 0),
                           x >= TINY]
            res_pc = np.zeros((10, 2))

            humanoid.set_dof_values(q[i])
            J_L = humanoid.compute_angular_momentum_jacobian(np.zeros(3))
            H_L = humanoid.compute_angular_momentum_hessian(np.zeros(3))
            for j in range(10):  # try 10 direction
                obj = cvx.Maximize(np.sin(float(j) / np.pi / 2) * u +
                                   np.sin(float(j) / np.pi / 2) * x)
                prob = cvx.Problem(obj, constraints)
                prob.solve(solver=cvx.ECOS, abstol=TINY)
                res_pc[j, :] = np.array([u.value, x.value])

            for u_, x_ in res_pc:
                qd_ = qs[i] * np.sqrt(x_)
                qdd_ = qs[i] * u_ + qss[i] * x_

                # gravito-inertia wrench
                J_COM = humanoid.compute_com_jacobian()
                H_COM = humanoid.compute_com_hessian()
                a_COM = np.dot(J_COM, qdd_) + np.dot(qd_, np.dot(H_COM, qd_))
                r_COM = humanoid.com
                Pdot_O = humanoid.mass * a_COM
                Ldot_O = np.dot(J_L, qdd_) + np.dot(qd_, np.dot(H_L, qd_))

                wcontact_O = np.zeros(6)
                wcontact_O[:3] = Pdot_O - humanoid.mass * g
                wcontact_O[3:] = Ldot_O - np.cross(r_COM, g) * humanoid.mass

                wstatic_O = np.r_[- humanoid.mass * g,
                                  - np.cross(r_COM, g) * humanoid.mass]

                eps = 0.1
                wsafe_O = wstatic_O * eps + wcontact_O * (1 - eps)

                # Consistency between topp matrices and actual dynamcis
                wc0 = (pc_contact.a[i] * u_ +
                       pc_contact.b[i] * x_ + pc_contact.c[i])
                F = stance.compute_wrench_face(np.zeros(3))
                wc1 = np.dot(F, wcontact_O)
                assert np.allclose(wc0, wc1, atol=TINY)

                # Consistency with pymanoid wrench computation
                wrenches = stance.find_supporting_wrenches(
                    wsafe_O, np.zeros(3))
                assert wrenches is not None
