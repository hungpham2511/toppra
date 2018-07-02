import toppra.solverwrapper.cy_seidel_solverwrapper as seidel
import numpy as np
import pytest

testdata = [
    (np.array([1.0, 2], dtype=float), np.array([]), np.array([]), -1.0, 1.0, 1, 3, 1, -2),
    (np.array([1.0, 2], dtype=float), [], [], -1.0, 1.0, 1, 3, 1, -2),
    (np.array([1.0, 2], dtype=float), None, None, -1.0, 1.0, 1, 3, 1, -2),
    (np.array([-2.0, 2], dtype=float), None, None, -1.0, 1.0, 1, 4, -1, -1),
    (np.array([1.0, 2], dtype=float), np.r_[4.0, -1.0], np.r_[-1.0, 0], -1.0, 1.0, 1, 2.25, 0.25, 0),
    (np.array([1.0, 0], dtype=float), np.r_[1.0, -1.0, -1, 1, 0, 0], np.r_[-1.5, -.5, -1.5, -0.5, 0, 0], -10000.0, 10000.0, 1, 0.5, 0.5, 3),
]

testids = [
    "boundonly1",
    "boundonly1a-emptylist",
    "boundonly1c-None",
    "boundonly2",
    "commoncase1",
    "bug(fixed)"
]


@pytest.mark.parametrize("v, a, b, low, high,"
                         "res_expected, optval_expected, optvar_expected, active_c_expected",
                         testdata, ids=testids)
def test_correct(v, a, b, low, high, res_expected, optval_expected, optvar_expected,
                 active_c_expected):
    data = seidel.solve_lp1d(v, a, b, low, high)
    res, optval, optvar, active_c = data

    assert res == res_expected
    assert optval == optval_expected
    assert optvar == optvar_expected
    assert active_c == active_c_expected


def test_infeasible():
    a = np.array([-1.0, 1.0])
    b = np.array([0.0, 0.5])
    data = seidel.solve_lp1d(np.r_[1.0, 2], a, b, -1, 1.00)
    res, optval, optvar, active_c = data

    assert res == 0

