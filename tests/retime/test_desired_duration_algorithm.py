import pytest
import numpy as np
import toppra
import toppra.constraint

toppra.setup_logging(level="INFO")


@pytest.fixture(params=[0, 1], name="velacc_fixture")
def velocity_acceleration_fixture(request):
    """ A pair of velocity, acceleration constraints.
    """
    np.random.seed(request.param)
    vvlim_ = 5 * np.random.rand(5) + 0.1
    valim_ = 20 * np.random.rand(5) + 0.1
    cvlim = toppra.constraint.JointVelocityConstraint(np.vstack((- vvlim_, vvlim_)).T)
    calim = toppra.constraint.JointAccelerationConstraint(np.vstack((- valim_, valim_)).T)
    yield cvlim, calim


@pytest.fixture(params=[1, 2])
def path_fixture(request):
    """ A random geometric path.
    """
    seed = request.param
    np.random.seed(seed)
    path = toppra.SplineInterpolator(np.linspace(0, 1, 5), np.random.randn(5, 5))
    yield path


def test_specific_duration_parametrization(velacc_fixture, path_fixture):
    """A simple test for `TOPPRAsd`, a variant of TOPPRA that computes
    parameterizations with specified duration.

    TOPPRAsd has an identical interface to TOPPRA with an additional
    method `set_desired_duration`.
    """
    t_desired = 50
    cvlim, calim = velacc_fixture
    instance = toppra.algorithm.TOPPRAsd([cvlim, calim], path_fixture)
    instance.set_desired_duration(t_desired)
    _, sd_vec, _ = instance.compute_parameterization(0, 0, atol=1e-3)

    t0 = 0.0
    for i in range(1, instance._N + 1):
        dti = 2 * (instance.gridpoints[i] - instance.gridpoints[i - 1]) / (sd_vec[i - 1] + sd_vec[i])
        t0 += dti
    np.testing.assert_allclose(t0, t_desired, atol=1e-3)


