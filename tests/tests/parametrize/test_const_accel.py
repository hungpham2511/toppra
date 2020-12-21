import toppra
import pytest
import numpy as np


@pytest.fixture(name='path')
def setup_geometric_path():
    yield toppra.SplineInterpolator([0, 1, 2], [(0, 0), (1, 2), (2, 0)])


def test_initialzie(path):
    gridpoints = [0, 0.5, 1, 1.5, 2]
    velocities = [1, 2, 2, 1, 0]
    # xd = [1, 4, 4, 1, 0]
    # ud = [6.0, 0, -6.0, -2.0]

    path_new = toppra.ParametrizeConstAccel(path, gridpoints, velocities)
    try:
        path_new.plot_parametrization(show=False)
    except Exception:
        # when run on CI, this fails because of some tkinter error
        pass

    assert path_new(0).shape == (2,)
    assert path_new([0]).shape == (1, 2)
    assert path_new.path_interval[1] > 0
    assert path_new.duration > 0


def test_evaluate_derivative(path):
    gridpoints = [0, 0.5, 1, 1.5, 2]
    velocities = [1, 2, 2, 1, 0]
    # xd = [1, 4, 4, 1, 0]
    # ud = [6.0, 0, -6.0, -2.0]

    path_new = toppra.ParametrizeConstAccel(path, gridpoints, velocities)

    path_new(np.r_[0, 0.1])
    path_new(np.r_[0, 0.1], 1)
    path_new(np.r_[0, 0.1], 2)
