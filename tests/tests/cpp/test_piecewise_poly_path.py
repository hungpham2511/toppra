import pytest
import toppra.cpp as tac
import numpy as np
import msgpack


pytestmark = pytest.mark.skipif(
    not tac.bindings_loaded(), reason="c++ bindings not built"
)


def test_load_cpp_bindings_ok():
    assert tac.bindings_loaded()


@pytest.fixture
def path():
    c = np.array(
        [[-0.500000, -0.500000, 1.500000, 0.500000, 0.000000, 3.000000, 0.000000, 0.000000,],
         [-0.500000, -0.500000, 0.000000, -1.000000, 1.500000, 2.500000, 1.000000, 3.000000,],
         [-0.500000, -0.500000, -1.500000, -2.500000, 0.000000, -1.000000, 2.000000, 4.000000,],]
    )
    c = c.reshape((3, 4, 2))
    p = tac.PiecewisePolyPath(c, [0, 1, 2, 3])
    yield p


def test_check_piecewise_poly_path(path):
    pos = path([0, 0.5, 1, 1.1, 2.5])
    np.testing.assert_allclose(pos[1], [0.3125, 1.5625])


def test_dof(path):
    assert path.dof == 2


def test_interval(path):
    np.testing.assert_allclose(path.path_interval, [0, 3])


def test_serialize(path):
    ss = path.serialize()
    path2 = tac.PiecewisePolyPath()
    path2.deserialize(ss)


def test_hermite():
    path = tac.PiecewisePolyPath.constructHermite(
        [[0, 0], [1, 1], [0, 0]], [[0, 0], [0, 0], [0, 0]], [1, 2, 3]
    )
    assert path.dof == 2
    np.testing.assert_allclose(path([1, 2, 3]), [[0, 0], [1, 1], [0, 0]])
