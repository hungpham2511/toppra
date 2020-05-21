import toppra.cpp as tac
import numpy as np


def test_load_cpp_bindings_ok():
    assert tac.bindings_loaded()


def test_check_piecewise_poly_path():
    c = np.array([
        [-0.500000, -0.500000, 1.500000, 0.500000, 0.000000, 3.000000, 0.000000, 0.000000],
        [-0.500000, -0.500000, 0.000000, -1.000000, 1.500000, 2.500000, 1.000000, 3.000000],
        [-0.500000, -0.500000, -1.500000, -2.500000, 0.000000, -1.000000, 2.000000, 4.000000]
    ])
    c = c.reshape((3, 4, 2))
    p = tac.PiecewisePolyPath(c, [0, 1, 2, 3])
    pos = p([0, 0.5, 1, 1.1, 2.5])
    np.testing.assert_allclose(pos[1], [0.3125, 1.5625])

    assert (p.dof == 2)


