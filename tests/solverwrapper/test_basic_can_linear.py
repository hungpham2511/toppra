"""A test suite for solverwrappers that implement solve methods for
canonical linear constraints. Wrapppers considered include:
'cvxpy', 'qpOASES', "ecos", 'hotqpOASES', 'seidel'.

"""
import pytest
import numpy as np
import numpy.testing as npt

import toppra
import toppra.constraint as constraint

toppra.setup_logging(level="INFO")

try:
    import cvxpy
    FOUND_CXPY = True
except ImportError:
    FOUND_CXPY = False

try:
    import mosek
    FOUND_MOSEK = True
except ImportError:
    FOUND_MOSEK = False


class RandomSecondOrderLinearConstraint(constraint.CanonicalLinearConstraint):
    """A random Second-Order non-identical constraint.

    This contraint is defined solely for testing purposes. It accepts
    a degree of freedom, then generates the coefficient randomly.

    """

    def __init__(self, dof, discretization_scheme=constraint.DiscretizationType.Collocation):
        super(RandomSecondOrderLinearConstraint, self).__init__()
        self.dof = dof
        self.set_discretization_type(discretization_scheme)
        self.identical = False
        self._format_string = "    Random Second-Order constraint (dof={:d}) \n".format(
            self.dof)

    def compute_constraint_params(self, path, gridpoints, scaling=1.0):
        assert scaling == 1.0, "In this mock class scaling needs to be 1"
        N = gridpoints.shape[0] - 1
        a = np.random.randn(N + 1, self.dof)
        b = np.random.randn(N + 1, self.dof)
        c = np.random.randn(N + 1, self.dof)
        F = np.random.randn(N + 1, self.dof, self.dof)
        g = np.random.rand(N + 1, self.dof)
        for i in range(N + 1):
            g[i] += F[i].dot(c[i])

        if self.discretization_type == constraint.DiscretizationType.Collocation:
            return a, b, c, F, g, None, None
        elif self.discretization_type == constraint.DiscretizationType.Interpolation:
            return constraint.canlinear_colloc_to_interpolate(
                a, b, c, F, g, None, None, gridpoints, identical=False)
        else:
            raise NotImplementedError("Other form of discretization not supported!")


@pytest.fixture(scope='class', params=['vel_accel'])
def basic_init_fixture(request):
    """ A fixture for testing basic capability of the solver wrapper.

    This test case has only two constraints, one velocity constraint
    and one acceleration constraint.
    """
    dof = 6
    np.random.seed(1)  # Use the same randomly generated way pts
    way_pts = np.random.randn(4, dof) * 0.6
    N = 200
    path = toppra.SplineInterpolator(np.linspace(0, 1, 4), way_pts)
    ss = np.linspace(0, 1, N + 1)
    # Velocity Constraint
    vlim_ = np.random.rand(dof) * 10 + 10
    vlim = np.vstack((-vlim_, vlim_)).T
    pc_vel = constraint.JointVelocityConstraint(vlim)
    # Acceleration Constraints
    alim_ = np.random.rand(dof) * 10 + 100
    alim = np.vstack((-alim_, alim_)).T
    pc_acc = constraint.JointAccelerationConstraint(alim)
    # random Second Order Constraint, only use for testing
    pc_rand = RandomSecondOrderLinearConstraint(dof)
    pcs = [pc_vel, pc_acc, pc_rand]
    yield pcs, path, ss, vlim, alim

    print("\n [TearDown] Finish PP Fixture")


@pytest.mark.parametrize("solver_name", ['cvxpy', 'qpOASES', "ecos", 'hotqpOASES', 'seidel'])
@pytest.mark.parametrize("i", [3, 10, 30])
@pytest.mark.parametrize("H", [np.array([[1.5, 0], [0, 1.0]]), np.zeros((2, 2)), None])
@pytest.mark.parametrize("g", [np.array([0.2, -1]), np.array([0.5, 1]), np.array([2.0, 1])])
@pytest.mark.parametrize("x_ineq", [(-1, 1), (0.2, 0.2), (0.4, 0.3), (np.nan, np.nan)])
@pytest.mark.skipif(not FOUND_CXPY, reason="This test requires cvxpy to validate results.")
def test_basic_init(basic_init_fixture, solver_name, i, H, g, x_ineq):
    """ A basic test case for wrappers.

    Notice that the input fixture `basic_init_fixture` is known to have two constraints,
    one velocity and one acceleration. Hence, in this test, I directly formulate
    an optimization with cvxpy to test the result.

    Parameters
    ----------
    basic_init_fixture: a fixture with only two constraints, one velocity and
        one acceleration constraint.

    """
    constraints, path, path_discretization, vlim, alim = basic_init_fixture
    if solver_name == "cvxpy":
        from toppra.solverwrapper.cvxpy_solverwrapper import cvxpyWrapper
        solver = cvxpyWrapper(constraints, path, path_discretization)
    elif solver_name == 'qpOASES':
        from toppra.solverwrapper.qpoases_solverwrapper import qpOASESSolverWrapper
        solver = qpOASESSolverWrapper(constraints, path, path_discretization)
    elif solver_name == 'hotqpOASES':
        from toppra.solverwrapper.hot_qpoases_solverwrapper import hotqpOASESSolverWrapper
        solver = hotqpOASESSolverWrapper(constraints, path, path_discretization)
    elif solver_name == 'ecos' and H is None:
        from toppra.solverwrapper.ecos_solverwrapper import ecosWrapper
        solver = ecosWrapper(constraints, path, path_discretization)
    elif solver_name == 'seidel' and H is None:
        from toppra.solverwrapper.cy_seidel_solverwrapper import seidelWrapper
        solver = seidelWrapper(constraints, path, path_discretization)
    else:
        return True  # Skip all other tests

    xmin, xmax = x_ineq
    xnext_min = 0
    xnext_max = 1

    # Results from solverwrapper to test
    solver.setup_solver()
    result_ = solver.solve_stagewise_optim(i - 2, H, g, xmin, xmax, xnext_min, xnext_max)
    result_ = solver.solve_stagewise_optim(i - 1, H, g, xmin, xmax, xnext_min, xnext_max)
    result = solver.solve_stagewise_optim(i, H, g, xmin, xmax, xnext_min, xnext_max)
    solver.close_solver()

    # Results from cvxpy, used as the actual, desired values
    ux = cvxpy.Variable(2)
    u = ux[0]
    x = ux[1]
    _, _, _, _, _, _, xbound = solver.params[0]  # vel constraint
    a, b, c, F, h, ubound, _ = solver.params[1]  # accel constraint
    a2, b2, c2, F2, h2, _, _ = solver.params[2]  # random constraint
    Di = path_discretization[i + 1] - path_discretization[i]
    v = a[i] * u + b[i] * x + c[i]
    v2 = a2[i] * u + b2[i] * x + c2[i]
    cvxpy_constraints = [
        x <= xbound[i, 1],
        x >= xbound[i, 0],
        F * v <= h,
        F2[i] * v2 <= h2[i],
        x + u * 2 * Di <= xnext_max,
        x + u * 2 * Di >= xnext_min,
    ]
    if not np.isnan(xmin):
        cvxpy_constraints.append(x <= xmax)
        cvxpy_constraints.append(x >= xmin)
    if H is not None:
        objective = cvxpy.Minimize(0.5 * cvxpy.quad_form(ux, H) + g * ux)
    else:
        objective = cvxpy.Minimize(g * ux)
    problem = cvxpy.Problem(objective, cvxpy_constraints)
    problem.solve(verbose=True)  # test with the same solver as cvxpywrapper
    if problem.status == "optimal":
        actual = np.array(ux.value).flatten()
        result = np.array(result).flatten()
        npt.assert_allclose(result, actual, atol=5e-3, rtol=1e-5)  # Very bad accuracy? why?
    else:
        assert np.all(np.isnan(result))


@pytest.mark.parametrize("solver_name", ['cvxpy', 'qpOASES', 'ecos', 'hotqpOASES', 'seidel'])
def test_infeasible_instance(basic_init_fixture, solver_name):
    """If the given parameters are infeasible, the solverwrapper should
    terminate gracefully and return a numpy vector [nan, nan].
    """
    constraints, path, path_discretization, vlim, alim = basic_init_fixture
    if solver_name == "cvxpy":
        from toppra.solverwrapper.cvxpy_solverwrapper import cvxpyWrapper
        solver = cvxpyWrapper(constraints, path, path_discretization)
    elif solver_name == 'qpOASES':
        from toppra.solverwrapper.qpoases_solverwrapper import qpOASESSolverWrapper
        solver = qpOASESSolverWrapper(constraints, path, path_discretization)
    elif solver_name == 'hotqpOASES':
        from toppra.solverwrapper.hot_qpoases_solverwrapper import hotqpOASESSolverWrapper
        solver = hotqpOASESSolverWrapper(constraints, path, path_discretization)
    elif solver_name == 'ecos':
        from toppra.solverwrapper.ecos_solverwrapper import ecosWrapper
        solver = ecosWrapper(constraints, path, path_discretization)
    elif solver_name == 'seidel':
        from toppra.solverwrapper.cy_seidel_solverwrapper import seidelWrapper
        solver = seidelWrapper(constraints, path, path_discretization)

    g = np.r_[0, 1].astype(float)

    solver.setup_solver()
    result = solver.solve_stagewise_optim(0, None, g, 1.1, 1.0, np.nan, np.nan)
    assert np.all(np.isnan(result))

    result = solver.solve_stagewise_optim(0, None, g, 1.1, 1.0, 0, -0.5)
    assert np.all(np.isnan(result))

    result = solver.solve_stagewise_optim(0, None, g, np.nan, np.nan, 0, -0.5)
    assert np.all(np.isnan(result))
    solver.close_solver()
