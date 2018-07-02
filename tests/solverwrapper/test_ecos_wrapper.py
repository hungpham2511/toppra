""" Tests for features specific to `ecosWrapper`.
"""
import pytest
import numpy as np
import toppra
import toppra.constraint as constraint
from .test_basic_conic_can_linear import path, vel_accel_robustaccel

toppra.setup_logging(level="DEBUG")

try:
    import mosek
    FOUND_MOSEK = True
except ImportError:
    FOUND_MOSEK = False

try:
    import cvxpy
    FOUND_CXPY = True
except ImportError:
    FOUND_CXPY = False


@pytest.mark.parametrize("i", [0, 5, 9])
@pytest.mark.parametrize("g", [np.array([0.2, -1]), np.array([0.5, 1]), np.array([2.0, 1])])
@pytest.mark.parametrize("x_ineq", [(-1, 1), (0.2, 0.2), (0.4, 0.3), (np.nan, np.nan)])
def test_linear_constraints_only(vel_accel_robustaccel, path, i, g, x_ineq):
    "Only canonical linear constraints."
    vel_c, acc_c, robust_acc_c = vel_accel_robustaccel
    path_dist = np.linspace(0, path.get_duration(), 10 + 1)
    solver = toppra.solverwrapper.ecosWrapper([vel_c, acc_c], path, path_dist)
    target_solver = toppra.solverwrapper.qpOASESSolverWrapper([vel_c, acc_c], path, path_dist)

    xmin, xmax = x_ineq
    xnext_min = 0
    xnext_max = 1

    result = solver.solve_stagewise_optim(i, None, g, xmin, xmax, xnext_min, xnext_max)
    target_result = target_solver.solve_stagewise_optim(i, None, g, xmin, xmax, xnext_min, xnext_max)

    np.testing.assert_allclose(result, target_result, atol=1e-5)
