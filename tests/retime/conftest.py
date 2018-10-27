import pytest
import numpy as np
import toppra
import toppra.constraint as constraint


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


