import pytest
import numpy as np

import toppra
import toppra.constraint as constraint

@pytest.fixture(params=[(0, 0)])
def vel_accel_robustaccel(request):
    "Velocity + Acceleration + Robust Acceleration constraint"
    dtype_a, dtype_ra = request.param
    vlims = np.array([[-1, 1], [-1, 2], [-1, 4]], dtype=float)
    alims = np.array([[-1, 1], [-1, 2], [-1, 4]], dtype=float)
    vel_cnst = constraint.JointVelocityConstraint(vlims)
    accl_cnst = constraint.JointAccelerationConstraint(alims, dtype_a)
    robust_accl_cnst = constraint.RobustCanonicalLinearConstraint(
        accl_cnst, [0.5, 0.1, 2.0], dtype_ra)
    yield vel_cnst, accl_cnst, robust_accl_cnst


@pytest.fixture
def path():
    np.random.seed(1)
    path = toppra.SplineInterpolator(np.linspace(0, 1, 5), np.random.randn(5, 3))
    yield path

