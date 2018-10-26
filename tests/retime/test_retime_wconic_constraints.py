import pytest
import numpy as np
import toppra
import toppra.constraint as constraint

toppra.setup_logging(level="INFO")

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


@pytest.mark.parametrize("solver_wrapper", ["cvxpy", "ecos"])
def test_toppra_conic(vel_accel_robustaccel, path, solver_wrapper):
    vel_c, acc_c, ro_acc_c = vel_accel_robustaccel
    acc_c.set_discretization_type(1)
    ro_acc_c.set_discretization_type(1)
    ro_instance = toppra.algorithm.TOPPRA([vel_c, ro_acc_c], path,
                                          solver_wrapper=solver_wrapper)

    X = ro_instance.compute_feasible_sets()
    assert np.all(X >= 0)
    assert not np.any(np.isnan(X))

    K = ro_instance.compute_controllable_sets(0, 0)
    assert np.all(K >= 0)
    assert not np.any(np.isnan(K))

    traj, _ = ro_instance.compute_trajectory(0, 0)
    assert traj is not None
    assert traj.get_duration() < 20 and traj.get_duration() > 0
