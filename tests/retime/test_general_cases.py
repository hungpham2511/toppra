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


@pytest.mark.parametrize("solver_wrapper", ["cvxpy", "qpoases", "hotqpoases", "seidel"])
def test_toppra_linear(basic_constraints, basic_path, solver_wrapper):
    """Solve some basic problem instances.

    Passing this test guaranetees that the basic functionalities are
    inplace.
    """
    vel_c, acc_c, ro_acc_c = basic_constraints
    instance = toppra.algorithm.TOPPRA(
        [vel_c, acc_c], basic_path, solver_wrapper=solver_wrapper)
    X = instance.compute_feasible_sets()
    assert np.all(X >= 0)
    assert not np.any(np.isnan(X))

    K = instance.compute_controllable_sets(0, 0)
    assert np.all(K >= 0)
    assert not np.any(np.isnan(K))

    traj, _ = instance.compute_trajectory(0, 0)
    assert traj is not None


@pytest.mark.parametrize("solver_wrapper", [
    "cvxpy,qpoases",
    "qpoases,hotqpoases",
    "qpoases,seidel",
    "hotqpoases,seidel"
])
def test_toppra_linear_compare(basic_constraints, basic_path, solver_wrapper):
    """ Compare the output of the algorithm for basic instances.

    """
    print("compare {:} and {:}".format(*solver_wrapper))
    solver_wrapper = solver_wrapper.split(",")
    vel_c, acc_c, ro_acc_c = basic_constraints
    instance = toppra.algorithm.TOPPRA(
        [vel_c, acc_c], basic_path, solver_wrapper=solver_wrapper[0])
    instance2 = toppra.algorithm.TOPPRA(
        [vel_c, acc_c], basic_path, solver_wrapper=solver_wrapper[1])

    K = instance.compute_controllable_sets(0, 0)
    K2 = instance2.compute_controllable_sets(0, 0)
    for i in range(instance._N, -1, -1):
        np.testing.assert_allclose(K[i], K2[i], atol=1e-6, rtol=1e-2,
                                   err_msg="Mismatched at i={:} / N={:}".format(i, instance._N))

    X = instance.compute_feasible_sets()
    X2 = instance2.compute_feasible_sets()
    for i in range(instance._N, -1, -1):
        np.testing.assert_allclose(X[i], X2[i], atol=1e-6, rtol=1e-2,
                                   err_msg="Mismatched at i={:} / N={:}".format(i, instance._N))

    sd, sdd, _ = instance.compute_parameterization(0, 0)
    sd2, sdd2, _ = instance2.compute_parameterization(0, 0)
    for i in range(instance._N - 1, -1, -1):
        np.testing.assert_allclose(sd[i], sd2[i], atol=1e-6, rtol=1e-2,
                                   err_msg="Mismatched at i={:} / N={:}".format(i, instance._N))

