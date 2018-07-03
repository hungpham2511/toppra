import pytest
import numpy as np
import numpy.testing as npt

import toppra
import toppra.constraint as constraint
from toppra.solverwrapper import (cvxpyWrapper, qpOASESSolverWrapper, ecosWrapper,
                                  hotqpOASESSolverWrapper, seidelWrapper)

toppra.setup_logging(level="DEBUG")

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


@pytest.fixture(scope='class', params=['vel_accel'])
def pp_fixture(request):
    """ Velocity & Acceleration Path Constraint.

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

    pcs = [pc_vel, pc_acc]
    yield pcs, path, ss, vlim, alim

    print "\n [TearDown] Finish PP Fixture"


@pytest.mark.parametrize("solver_name", ['cvxpy', 'qpOASES', "ecos", 'hotqpOASES', 'seidel'])
@pytest.mark.parametrize("i", [3, 10, 30])
@pytest.mark.parametrize("H", [np.array([[1.5, 0], [0, 1.0]]), np.zeros((2, 2)), None])
@pytest.mark.parametrize("g", [np.array([0.2, -1]), np.array([0.5, 1]), np.array([2.0, 1])])
@pytest.mark.parametrize("x_ineq", [(-1, 1), (0.2, 0.2), (0.4, 0.3), (np.nan, np.nan)])
@pytest.mark.skipif(not FOUND_CXPY, reason="This test requires cvxpy to validate results.")
def test_basic_init(pp_fixture, solver_name, i, H, g, x_ineq):
    """ A basic test case for wrappers.

    Notice that the input fixture `pp_fixture` is known to have two constraints,
    one velocity and one acceleration. Hence, in this test, I directly formulate
    an optimization with cvxpy to test the result.

    Parameters
    ----------
    pp_fixture: a fixture with only two constraints, one velocity and
        one acceleration constraint.

    """
    constraints, path, path_discretization, vlim, alim = pp_fixture
    if solver_name == "cvxpy":
        solver = cvxpyWrapper(constraints, path, path_discretization)
    elif solver_name == 'qpOASES':
        solver = qpOASESSolverWrapper(constraints, path, path_discretization)
    elif solver_name == 'hotqpOASES':
        solver = hotqpOASESSolverWrapper(constraints, path, path_discretization)
    elif solver_name == 'ecos' and H is None:
        solver = ecosWrapper(constraints, path, path_discretization)
    elif solver_name == 'seidel' and H is None:
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
    _, _, _, _, _, _, xbound = solver.params[0]
    a, b, c, F, h, ubound, _ = solver.params[1]
    Di = path_discretization[i + 1] - path_discretization[i]
    v = a[i] * u + b[i] * x + c[i]
    cvxpy_constraints = [
        u <= ubound[i, 1],
        u >= ubound[i, 0],
        x <= xbound[i, 1],
        x >= xbound[i, 0],
        F * v <= h,
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
    if FOUND_MOSEK:
        problem.solve(solver="MOSEK", verbose=True)
    else:
        problem.solve(solver="ECOS", verbose=True)
    if problem.status == "optimal":
        actual = np.array(ux.value).flatten()
        result = np.array(result).flatten()
        npt.assert_allclose(result, actual, atol=5e-3, rtol=1e-5)  # Very bad accuracy? why?
    else:
        assert np.all(np.isnan(result))


@pytest.mark.parametrize("solver_name", ['cvxpy', 'qpOASES', 'ecos', 'hotqpOASES', 'seidel'])
def test_infeasible_instance(pp_fixture, solver_name):
    """If the given parameters are infeasible, the solverwrapper should
    terminate gracefully and return a numpy vector [nan, nan].
    """
    constraints, path, path_discretization, vlim, alim = pp_fixture
    if solver_name == "cvxpy":
        solver = cvxpyWrapper(constraints, path, path_discretization)
    elif solver_name == 'qpOASES':
        solver = qpOASESSolverWrapper(constraints, path, path_discretization)
    elif solver_name == 'hotqpOASES':
        solver = hotqpOASESSolverWrapper(constraints, path, path_discretization)
    elif solver_name == 'ecos':
        solver = ecosWrapper(constraints, path, path_discretization)
    elif solver_name == 'seidel':
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
