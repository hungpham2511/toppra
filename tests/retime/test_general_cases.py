"""General cases

This test suite guarantees that the basic functionalities provided by
toppra and the many interfaces to optimization solvers contained with
it work.
"""
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
def vel_accel_constraints(request):
    "Velocity + Acceleration + Robust Acceleration constraint"
    dtype_a, dtype_ra = request.param
    vlims = np.array([[-1, 1], [-1, 2], [-1, 4], [-3, 4], [-2, 4], [-3, 4], [-2, 5]],
                     dtype=float)
    alims = np.array([[-1, 1], [-1, 2], [-1, 4], [-3, 4], [-2, 4], [-3, 4], [-2, 5]],
                     dtype=float)
    vel_cnst = constraint.JointVelocityConstraint(vlims)
    accl_cnst = constraint.JointAccelerationConstraint(alims, dtype_a)
    robust_accl_cnst = constraint.RobustCanonicalLinearConstraint(
        accl_cnst, [1e-4, 1e-4, 5e-4], dtype_ra)
    yield vel_cnst, accl_cnst, robust_accl_cnst


@pytest.fixture(params=[1, 2])
def path(request):
    seed = request.param
    np.random.seed(seed)
    path = toppra.SplineInterpolator(np.linspace(0, 1, 5), np.random.randn(5, 7))
    yield path


@pytest.mark.parametrize("solver_wrapper", ["cvxpy", "qpoases", "hotqpoases", "seidel"])
def test_toppra_linear(vel_accel_constraints, path, solver_wrapper):
    """Basic problem instance.

    Only check for validity of the feasible sets, controllable sets.

    """
    vel_c, acc_c, ro_acc_c = vel_accel_constraints
    instance = toppra.algorithm.TOPPRA([vel_c, acc_c], path, solver_wrapper=solver_wrapper)
    X = instance.compute_feasible_sets()
    assert np.all(X >= 0)
    assert not np.any(np.isnan(X))

    K = instance.compute_controllable_sets(0, 0)
    assert np.all(K >= 0)
    assert not np.any(np.isnan(K))

    traj, _ = instance.compute_trajectory(0, 0)
    assert traj is not None


@pytest.mark.parametrize("solver_wrapper", [
    ("cvxpy", "qpoases"),
    ("qpoases", "hotqpoases"),
    ("qpoases", "ecos"),
    ("qpoases", "seidel")
])
def test_toppra_linear_compare(vel_accel_constraints, path, solver_wrapper):
    """ Compare the output of the algorithm
    """
    print("compare {:} and {:}".format(*solver_wrapper))
    vel_c, acc_c, ro_acc_c = vel_accel_constraints
    instance = toppra.algorithm.TOPPRA([vel_c, acc_c], path, solver_wrapper=solver_wrapper[0])
    instance2 = toppra.algorithm.TOPPRA([vel_c, acc_c], path, solver_wrapper=solver_wrapper[1])

    K = instance.compute_controllable_sets(0, 0)
    K2 = instance2.compute_controllable_sets(0, 0)
    for i in range(instance._N, -1, -1):
        np.testing.assert_allclose(K[i], K2[i], atol=1e-6,
                                   err_msg="Mismatched at i={:} / N={:}".format(i, instance._N))

    X = instance.compute_feasible_sets()
    X2 = instance2.compute_feasible_sets()
    for i in range(instance._N, -1, -1):
        np.testing.assert_allclose(X[i], X2[i], atol=1e-6,
                                   err_msg="Mismatched at i={:} / N={:}".format(i, instance._N))

    sd, sdd, _ = instance.compute_parameterization(0, 0)
    sd2, sdd2, _ = instance2.compute_parameterization(0, 0)
    for i in range(instance._N - 1, -1, -1):
        np.testing.assert_allclose(sd[i], sd2[i], atol=1e-6,
                                   err_msg="Mismatched at i={:} / N={:}".format(i, instance._N))

