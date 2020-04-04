import toppra
import pytest
import numpy as np
from numpy.testing import assert_allclose


@pytest.fixture(name="f_scalar")
def give_a_simple_scalar_function_without_derivatives():
    f = toppra.SimplePath([0, 1, 2], [0, 1, 1])
    yield f


def test_check_scalar_value_same_as_endpoint(f_scalar):
    assert f_scalar(1) == 1.0
    assert f_scalar(2) == 1.0


def test_first_derivative_is_continuous(f_scalar):
    d1 = f_scalar(np.linspace(0, 2, 200), 1)
    max_d1 = np.max(np.abs(np.diff(d1)))
    assert max_d1 < 0.1


def test_first_derivative_midpoint_is_average_between_two_points(f_scalar):
    assert_allclose(f_scalar(1, 1), np.array(0.5))


def test_first_derivative_zero_at_endpoints(f_scalar):
    assert_allclose(f_scalar(0, 1), np.array(0))
    assert_allclose(f_scalar(2, 1), np.array(0))

