import pytest
import toppra as fa
from toppra import (qpOASESPPSolver, TINY, SplineInterpolator)
from testingUtils import canonical_to_TypeI
import numpy as np
import openravepy as orpy
import cvxpy as cvx
# from test_Path_Constraints import pymanoid_fixture

VERBOSE = False
DEBUG = False


@pytest.fixture(scope='class',
                params=['vel_accel', 'vel_torque', 'vel_accel_torque'])
def pp_fixture(request):
    """ Velocity & Acceleration Path Constraint
    """
    env = orpy.Environment()
    env.Load('robots/pumaarm.zae')
    # env.SetViewer('qtosg')
    robot = env.GetRobots()[0]
    robot.SetDOFTorqueLimits(np.ones(robot.GetDOF()) * 100)
    dof = robot.GetDOF()
    np.random.seed(1)  # Use the same randomly generated way pts
    way_pts = np.random.randn(4, dof) * 0.6
    N = 200
    pi = SplineInterpolator(np.linspace(0, 1, 4), way_pts)
    ss = np.linspace(0, 1, N + 1)
    # Velocity Constraint
    vlim_ = np.random.rand(dof) * 10 + 10
    vlim = np.vstack((-vlim_, vlim_)).T
    pc_vel = fa.create_velocity_path_constraint(pi, ss, vlim)
    # Acceleration Constraints
    alim_ = np.random.rand(dof) * 10 + 100
    alim = np.vstack((-alim_, alim_)).T
    pc_acc = fa.create_acceleration_path_constraint(pi, ss, alim)
    # Torque constraints

    pc_torque = fa.create_rave_torque_path_constraint(pi, ss, robot)
    if request.param == 'vel_accel':
        pcs = [pc_vel, pc_acc]
        yield pcs, qpOASESPPSolver(pcs, verbose=VERBOSE)
    elif request.param == 'vel_torque':
        pcs = [pc_vel, pc_torque]
        yield pcs, qpOASESPPSolver(pcs)
    elif request.param == 'vel_accel_torque':
        pcs = [pc_vel, pc_acc, pc_torque]
        yield pcs, qpOASESPPSolver(pcs)
    else:
        pcs = [pc_vel, pc_acc, pc_torque, canonical_to_TypeI(pc_acc)]
        yield pcs, qpOASESPPSolver(pcs, verbose=VERBOSE)
    print "\n [TearDown] Finish PP Fixture"


class TestFunc_QpoasesPPsolver_micro(object):
    """Test suite for micro functions of the QPOASES PathParameterization
    solver.
    """

    def test_micro_func_one_step(self, pp_fixture):
        """ Test one-step set
        """
        pcs, pp = pp_fixture

        x = cvx.Variable()
        u = cvx.Variable()
        pp.reset_operational_rows()  # A single reset should suffice
        pp.nWSR_up = np.ones((pp.N + 1, 1), dtype=int) * pp.nWSR_cnst
        pp.nWSR_down = np.ones((pp.N + 1, 1), dtype=int) * pp.nWSR_cnst
        for i in range(5, 10):
            ds = pp.ss[i + 1] - pp.ss[i]
            xmin = 0
            xmax = 1
            init = (True if i == 5 else False)  # i = 6,...10, use hotstart
            K_i_low, K_i_high = pp.one_step(i, xmin, xmax, init=init)

            constraints = [x >= 0, x + 2 * ds * u >= xmin,
                           x + 2 * ds * u <= xmax]
            for pc in pcs:
                if pc.nm != 0:
                    constraints.append(
                        pc.a[i] * u + pc.b[i] * x + pc.c[i] <= 0)
                if pc.nv != 0:
                    v = cvx.Variable(pc.nv)
                    constraints.append(
                        (pc.abar[i] * u + pc.bbar[i] * x + pc.cbar[i] ==
                         pc.D[i] * v),
                        pc.l[i] <= v, v <= pc.h[i])
                    if pc.niq != 0:
                        constraints.append(
                            pc.lG[i] <= pc.G[i] * v,
                            pc.hG[i] >= pc.G[i] * v,
                        )
            obj = cvx.Maximize(x)
            prob = cvx.Problem(obj, constraints)
            prob.solve(solver=cvx.CVXOPT)
            assert np.allclose(x.value, K_i_high)
            obj = cvx.Minimize(x)
            prob = cvx.Problem(obj, constraints)
            prob.solve(solver=cvx.CVXOPT)
            assert np.allclose(x.value, K_i_low)

    def test_micro_func_reach(self, pp_fixture):
        """ Test one-step set
        """
        pcs, pp = pp_fixture

        x = cvx.Variable()
        u = cvx.Variable()

        pp.reset_operational_rows()  # A single reset should suffice
        for i in range(5, 10):
            ds = pp.ss[i+1] - pp.ss[i]
            xmin = 0
            xmax = 1
            init = (True if i == 5 else False)  # i = 6,...10, use hotstart
            xmin_i, xmax_i = pp.reach(i, xmin, xmax, init=init)

            constraints = [x >= 0, x >= xmin, x <= xmax]
            for pc in pcs:
                if pc.nm != 0:
                    constraints.append(
                        pc.a[i] * u + pc.b[i] * x + pc.c[i] <= 0)
                if pc.nv != 0:
                    v = cvx.Variable(pc.nv)
                    constraints.append(
                        (pc.abar[i] * u + pc.bbar[i] * x + pc.cbar[i] ==
                         pc.D[i] * v),
                        pc.l[i] <= v, v <= pc.h[i])
                    if pc.niq != 0:
                        constraints.append(
                            pc.lG[i] <= pc.G[i] * v,
                            pc.hG[i] >= pc.G[i] * v,
                        )
            obj = cvx.Maximize(x + 2 * ds * u)
            prob = cvx.Problem(obj, constraints)
            prob.solve(solver=cvx.CVXOPT)
            assert np.allclose(x.value + 2 * ds * u.value, xmax_i)
            obj = cvx.Minimize(x + 2 * ds * u)
            prob = cvx.Problem(obj, constraints)
            prob.solve(solver=cvx.CVXOPT)
            assert np.allclose(x.value + 2 * ds * u.value, xmin_i)

    def test_micro_func_proj_x(self, pp_fixture):
        """ Projection of x backto some interval
        """
        pcs, pp = pp_fixture

        x = cvx.Variable()
        u = cvx.Variable()

        pp.reset_operational_rows()  # A single reset should suffice
        for i in range(5, 10):
            xmin = 0
            xmax = 1
            init = (True if i == 5 else False)  # i = 6,...10, use hotstart
            xmin_i, xmax_i = pp.proj_x_admissible(i, xmin, xmax, init=init)

            constraints = [x >= 0, x >= xmin, x <= xmax]
            for pc in pcs:
                if pc.nm != 0:
                    constraints.append(
                        pc.a[i] * u + pc.b[i] * x + pc.c[i] <= 0)
                if pc.nv != 0:
                    v = cvx.Variable(pc.nv)
                    constraints.append(
                        (pc.abar[i] * u + pc.bbar[i] * x + pc.cbar[i] ==
                         pc.D[i] * v),
                        pc.l[i] <= v, v <= pc.h[i])
                    if pc.niq != 0:
                        constraints.append(
                            pc.lG[i] <= pc.G[i] * v,
                            pc.hG[i] >= pc.G[i] * v,
                        )
            obj = cvx.Maximize(x)
            prob = cvx.Problem(obj, constraints)
            prob.solve(solver=cvx.CVXOPT)
            assert np.allclose(x.value, xmax_i)
            obj = cvx.Minimize(x)
            prob = cvx.Problem(obj, constraints)
            prob.solve(solver=cvx.CVXOPT)
            assert np.allclose(x.value, xmin_i)

    def test_greedy_step(self, pp_fixture):
        """ test get_u_max function
        """
        pcs, pp = pp_fixture

        u = cvx.Variable()
        x = cvx.Variable()
        # Setup matrices
        for reg in np.linspace(0, 1., 4):
            pp.reset_operational_rows()  # A single reset should suffice
            pp.A[:, 0, 1] = 1.
            pp.A[:, 0, 0] = 0.
            pp.A[:, 1, 1] = 1.
            pp.A[:pp.N, 1, 0] = 2 * (pp.ss[1:] - pp.ss[:-1])
            pp.nWSR_topp = np.ones((pp.N + 1, 1), dtype=int) * pp.nWSR_cnst
            reg = 0.
            for i in range(5, 10):
                x_cur = 0.4
                xmin = 0.1
                xmax = 0.5
                init = (True if i == 5 else False)  # i = 6,...10, use hotstart
                u_, x_ = pp.greedy_step(i, x_cur, xmin, xmax, init=init, reg=reg)
                ds = pp.ss[i + 1] - pp.ss[i]

                constraints = [x + 2 * ds * u >= xmin,
                               x + 2 * ds * u <= xmax,
                               x == x_cur]
                obj_sum = u
                for pc in pcs:
                    if pc.nm != 0:
                        constraints.append(
                            pc.a[i] * u + pc.b[i] * x + pc.c[i] <= 0)
                    if pc.nv != 0:
                        v = cvx.Variable(pc.nv)
                        obj_sum += reg * cvx.norm2(v) ** 2
                        constraints.append(
                            (pc.abar[i] * u + pc.bbar[i] * x + pc.cbar[i] ==
                             pc.D[i] * v),
                            pc.l[i] <= v, v <= pc.h[i])
                        if pc.niq != 0:
                            constraints.append(
                                pc.lG[i] <= pc.G[i] * v,
                                pc.hG[i] >= pc.G[i] * v,
                            )

                obj = cvx.Maximize(obj_sum)
                prob = cvx.Problem(obj, constraints)
                prob.solve(solver=cvx.CVXOPT)
                assert np.allclose(u.value, u_)
                x__ = x.value + 2 * ds * u.value
                assert np.allclose(x__, x_)


class Test_QpoasesPPsolver_main_funcs(object):
    """ Very basic check
    """

    def test_comp_controllable(self, pp_fixture):
        """
        """
        pcs, solver = pp_fixture
        I0 = np.r_[0.1, 0.2]
        IN = np.r_[0.1, 0.2]
        solver.set_start_interval(I0)
        solver.set_goal_interval(IN)
        # Basic checks
        res_ctrl = solver.solve_controllable_sets()
        assert res_ctrl is True
        ctrl_sets = solver.K
        for i in range(solver.N+1):
            assert ctrl_sets[i, 1] >= ctrl_sets[i, 0]
            assert ctrl_sets[i, 0] >= 0
        assert ctrl_sets[solver.N, 0] >= IN[0] - TINY
        assert ctrl_sets[solver.N, 1] <= IN[1] + TINY

    def test_comp_reachable(self, pp_fixture):
        pcs, solver = pp_fixture
        I0 = np.r_[0.1, 0.2]
        IN = np.r_[0.1, 0.2]
        solver.set_start_interval(I0)
        solver.set_goal_interval(IN)
        # Basic checks
        res_rch = solver.solve_reachable_sets()
        assert res_rch is True
        rch_sets = solver.L
        for i in range(solver.N+1):
            assert rch_sets[i, 1] >= rch_sets[i, 0] - TINY
            assert rch_sets[i, 0] >= - TINY
        assert rch_sets[0, 0] >= I0[0] - TINY
        assert rch_sets[0, 1] <= I0[1] + TINY

    def test_comp_topp(self, pp_fixture):
        pcs, solver = pp_fixture
        I0 = np.r_[0.1, 0.2]
        IN = np.r_[0.1, 0.2]
        solver.set_start_interval(I0)
        solver.set_goal_interval(IN)
        us, xs = solver.solve_topp(reg=0)

        # Proper parameteriation
        assert xs[0] <= I0[1] + TINY and xs[0] >= I0[0] - TINY
        assert xs[-1] <= IN[1] + TINY and xs[-1] >= IN[0] - TINY
        assert np.all(xs >= 0)
        for i in range(solver.N):
            Delta_i = solver.ss[i+1] - solver.ss[i]
            assert np.allclose(xs[i+1], xs[i] + 2 * us[i] * Delta_i)

        # Constraint satisfy-ability
        for i in range(solver.N):
            for c in pcs:
                if c.nm != 0:
                    assert np.all(
                        c.a[i] * us[i] + c.b[i] * xs[i] + c.c[i] <= TINY)

    def test_comp_topp_single_float(self, pp_fixture):
        """Test :func:`solve_topp` when starting and goal velocities are
        floats.

        """
        pcs, solver = pp_fixture
        x_init = 0
        x_goal = 0.1
        solver.set_start_interval(x_init)
        solver.set_goal_interval(x_goal)
        us, xs = solver.solve_topp(reg=0)

        # Proper parameteriation
        assert xs[0] <= x_init + TINY and xs[0] >= x_init - TINY
        assert xs[-1] <= x_goal + TINY and xs[-1] >= x_goal - TINY
        assert np.all(xs >= 0)
        for i in range(solver.N):
            Delta_i = solver.ss[i + 1] - solver.ss[i]
            assert np.allclose(xs[i + 1], xs[i] + 2 * us[i] * Delta_i)

        # Constraint satisfy-ability
        for i in range(solver.N):
            for c in pcs:
                if c.nm != 0:
                    assert np.all(
                        c.a[i] * us[i] + c.b[i] * xs[i] + c.c[i] <= TINY)



