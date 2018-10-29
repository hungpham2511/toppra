"""This test suite ensures that problem scaling does not affect
solution quality, in the sense that a very short path (~1e-2) and a
very long path (~1e2) can be both parameterized.
"""
import pytest
import numpy as np
import toppra
import toppra.constraint as constraint

toppra.setup_logging(level="INFO")


@pytest.mark.parametrize("solver_wrapper", ["ecos", "hotqpoases", "seidel"])
def test_linear_case1(vel_accel_constraints, path, solver_wrapper):
    """A generic test case.

    Compare scaling between 0.5 and 1.0. Since the scaling factor is
    quite small, resulting trajectories should have similar durations.

    """
    vel_c, acc_c, ro_acc_c = vel_accel_constraints
    instance_scale1 = toppra.algorithm.TOPPRA(
        [vel_c, acc_c], path, solver_wrapper=solver_wrapper, scaling=1.0)
    instance_scale05 = toppra.algorithm.TOPPRA(
        [vel_c, acc_c], path, solver_wrapper=solver_wrapper, scaling=5)
    traj1, _ = instance_scale1.compute_trajectory()
    traj05, _ = instance_scale05.compute_trajectory()
    # accurate up to 0.1%
    np.testing.assert_allclose(traj1.duration, traj05.duration, rtol=1e-3)
