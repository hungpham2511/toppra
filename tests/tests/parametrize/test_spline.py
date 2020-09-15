import toppra
import pytest


@pytest.fixture(name='path')
def setup_geometric_path():
    yield toppra.SplineInterpolator([0, 1, 2], [(0, 0), (1, 2), (2, 0)])


def test_initialzie(path):
    gridpoints = [0, 0.5, 1, 1.5, 2]
    velocities = [1, 2, 2, 1, 0]
    # xd = [1, 4, 4, 1, 0]
    # ud = [6.0, 0, -6.0, -2.0]

    path_new = toppra.ParametrizeSpline(path, gridpoints, velocities)
    assert path_new.path_interval[0] == 0
    assert path_new.path_interval[-1] > 0
