import toppra
import pytest
import numpy as np
from numpy.testing import assert_allclose

@pytest.fixture(name="f_2dof")
def give_a_2dof_function_without_derivatives():
    f = toppra.SimplePath([0, 1, 2], np.array([[0, 0], [1, 2.0], [1, 2.0]]))
    yield f

def test_check_basic_2dof_shape(f_2dof):
    d1 = f_2dof(np.linspace(0, 2, 200), 1)
    assert d1.shape == (200, 2)

@pytest.fixture(name="f_scalar")
def give_a_simple_scalar_function_without_derivatives():
    f = toppra.SimplePath([0, 1, 2], np.array([0, 1, 1]))
    yield f


def test_check_scalar_value_same_as_endpoint(f_scalar):
    assert f_scalar(1) == 1.0
    assert f_scalar(2) == 1.0


def test_first_derivative_is_continuous(f_scalar):
    d1 = f_scalar(np.linspace(0, 2, 200), 1)
    max_d1 = np.max(np.abs(np.diff(d1, axis=0)))
    assert max_d1 < 0.1


def test_first_derivative_midpoint_is_average_between_two_points(f_scalar):
    assert_allclose(f_scalar(1, 1), np.array(0.5))


def test_first_derivative_zero_at_endpoints(f_scalar):
    assert_allclose(f_scalar(0, 1), np.array(0))
    assert_allclose(f_scalar(2, 1), np.array(0))


@pytest.fixture(name="f_scalar_wd")
def given_a_simple_scalar_function_with_first_derivatives():
    f = toppra.SimplePath([0, 1, 2], np.array([0, 1, 1]), np.array([0, 2, 0]))
    yield f


def test_correct_derivatives_as_specified(f_scalar_wd):
    assert_allclose(f_scalar_wd(1, 1), 2.0)
    assert_allclose(f_scalar_wd(0, 1), 0)
    assert_allclose(f_scalar_wd(2, 1), 0)


@pytest.fixture(name="fm")
def given_a_vector_path_without_specified_derivative():
    yield toppra.SimplePath([0, 1, 2], np.array([[0, 0], [1, 2], [1, 2]]))


def test_path_interpolate_are_correct(fm):
    assert fm(0.5).shape == (2,)
    assert isinstance(fm(0.5)[0], float)
    assert_allclose(fm(0), [0, 0])
    assert_allclose(fm(1), [1, 2])


def test_dof_is_correct(fm):
    assert fm.dof == 2


def test_path_interval_is_correct(fm):
    assert_allclose(fm.path_interval, np.array([0.0, 2.0]))
