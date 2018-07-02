import pytest
import numpy as np
import toppra
import toppra.constraint as constraint
import matplotlib.pyplot as plt

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


@pytest.fixture(params=[(0, 0)])
def vel_accel_robustaccel(request):
    "Velocity + Acceleration + Robust Acceleration constraint"
    dtype_a, dtype_ra = request.param
    vlims = np.array([[-1, 1], [-1, 2], [-1, 4]], dtype=float)
    alims = np.array([[-1, 1], [-1, 2], [-1, 4]], dtype=float)
    vel_cnst = constraint.JointVelocityConstraint(vlims)
    accl_cnst = constraint.JointAccelerationConstraint(alims, dtype_a)
    robust_accl_cnst = constraint.RobustCanonicalLinearConstraint(
        accl_cnst, [1e-4, 1e-4, 5e-4], dtype_ra)
    yield vel_cnst, accl_cnst, robust_accl_cnst


@pytest.fixture(params=[1, 2])
def path(request):
    seed = request.param
    np.random.seed(seed)
    path = toppra.SplineInterpolator(np.linspace(0, 1, 5), np.random.randn(5, 3))
    yield path


@pytest.mark.parametrize("solver_wrapper", ["cvxpy", "ecos", "qpoases", "hotqpoases", "seidel"])
def test_toppra_linear(vel_accel_robustaccel, path, solver_wrapper):
    vel_c, acc_c, ro_acc_c = vel_accel_robustaccel
    instance = toppra.algorithm.TOPPRA([vel_c, acc_c], path, solver_wrapper=solver_wrapper)
    X = instance.compute_feasible_sets()
    assert np.all(X >= 0)
    assert not np.any(np.isnan(X))

    K = instance.compute_controllable_sets(0, 0)
    assert np.all(K >= 0)
    assert not np.any(np.isnan(K))

    traj, _ = instance.compute_trajectory(0, 0)
    assert traj is not None


@pytest.mark.parametrize("solver_wrapper", ["cvxpy", "ecos"])
def test_toppra_conic(vel_accel_robustaccel, path, solver_wrapper):
    vel_c, acc_c, ro_acc_c = vel_accel_robustaccel
    acc_c.set_discretization_type(1)
    ro_acc_c.set_discretization_type(1)
    ro_instance = toppra.algorithm.TOPPRA([vel_c, ro_acc_c], path, solver_wrapper=solver_wrapper)

    X = ro_instance.compute_feasible_sets()
    assert np.all(X >= 0)
    assert not np.any(np.isnan(X))

    K = ro_instance.compute_controllable_sets(0, 0)
    assert np.all(K >= 0)
    assert not np.any(np.isnan(K))

    traj, _ = ro_instance.compute_trajectory(0, 0)
    assert traj is not None
    assert traj.get_duration() < 20 and traj.get_duration() > 0



@pytest.mark.parametrize("solver_wrapper", [("cvxpy", "qpoases"),
                                            ("cvxpy", "hotqpoases"),
                                            ("cvxpy", "ecos"),
                                            ("cvxpy", "seidel")])
def test_toppra_linear_compare(vel_accel_robustaccel, path, solver_wrapper):
    "Compare the output of the algorithm"
    vel_c, acc_c, ro_acc_c = vel_accel_robustaccel
    instance = toppra.algorithm.TOPPRA([vel_c, acc_c], path, solver_wrapper=solver_wrapper[0])
    instance2 = toppra.algorithm.TOPPRA([vel_c, acc_c], path, solver_wrapper=solver_wrapper[1])

    X = instance.compute_feasible_sets()
    X2 = instance2.compute_feasible_sets()
    np.testing.assert_allclose(X, X2, atol=1e-5)

    sd, sdd, _ = instance.compute_parameterization(0, 0)
    sd2, sdd2, _ = instance2.compute_parameterization(0, 0)
    np.testing.assert_allclose(X, X2, atol=1e-5)
