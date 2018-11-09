import toppra.solverwrapper.cy_seidel_solverwrapper as seidel
import numpy as np
from numpy import array
import pytest
import cvxpy as cvx

testdata_correct = [
    ([1, 2, 3.0], None, None, None, [-1, -1], [1, 1], [-1, 1],
     1, 6, [1, 1], [-2, -4]),
    ([-2, 2, 2.0], None, None, None, [-1, -1], [1, 1], [-1, 1],
     1, 6, [-1, 1], [-1, -4]),
    ([1, 2, 3], (1, -1), (1, 1), (-1, -0.5), [-1, -1], [1, 1], [-1, -1],
     1, 4.75, [0.25, 0.75], [0, 1]),
    ([-1, 0.01, 0], (1, -1), (1, 1), (-1, -0.5), [-1, -1], [1, 1], [-1, -1],
     1, 0.995, [-1, -0.5], [-1, 1]),
    ([1, 2, 0],
     (1.36866544,  1.28199038, -0.19515422,  0.97578149,  0.64391477,
      -0.0811908 , -0.70696349, -1.01804875,  0.5742392 ,  0.02939029),
     ( 0.1969094 ,  1.13910161,  0.10109674,  1.71246466, -0.45206747,
       -0.51302219, -1.16558797,  0.19919171, -0.906885  ,  0.94722345),
     (-2.68926068, -1.59762444, -2.03337493, -2.04617298, -1.09241401,
      -1.67319798, -1.9483617 , -1.57529407, -1.37795315, -3.47919232), [-100, -100], [100, 100],
     [0, 1], 1, 2.5547484757095305, [-1.18181729266432, 1.8682828841869252], [3, 7]),
    ([1, 2, 0],
     (1.36866544,  1.28199038, -0.19515422,  0.97578149,  0.64391477,
      -0.0811908 , -0.70696349, -1.01804875,  0.5742392 ,  0.02939029),
     ( 0.1969094 ,  1.13910161,  0.10109674,  1.71246466, -0.45206747,
       -0.51302219, -1.16558797,  0.19919171, -0.906885  ,  0.94722345),
     (-2.68926068, -1.59762444, -2.03337493, -2.04617298, -1.09241401,
      -1.67319798, -1.9483617 , -1.57529407, -1.37795315, -3.47919232), [-100, -100], [100, 100],
     [5, 9], 1, 2.5547484757095305, [-1.18181729266432, 1.8682828841869252], [3, 7]),
    ([1, 2, 0], [-0.01, 0.01], [-1, 1], [0, 0.5], [-1, -1], [1, 1], [0, 1], 0, None, None, None)
]

testids_correct = [
    "fixbound1",
    "fixbound2",
    "two_constraints",
    "two_constraints",
    "random_10_c_warms",
    "random_10_c_warms",
    "bug"
]


@pytest.mark.parametrize("v, a, b, c, low, high, active_c,"
                         "res_expected, optval_expected, optvar_expected, active_c_expected",
                         testdata_correct, ids=testids_correct)
def test_correct(v, a, b, c, low, high, active_c, res_expected, optval_expected,
                 optvar_expected, active_c_expected):
    if a is None:
        a_np = None
        b_np = None
        c_np = None
    else:
        a_np = np.array(a, dtype=float)
        b_np = np.array(b, dtype=float)
        c_np = np.array(c, dtype=float)
    data = seidel.solve_lp2d(np.array(v, dtype=float), a_np, b_np, c_np,
                                     np.array(low, dtype=float), np.array(high, dtype=float), np.array(active_c, dtype=int))
    res, optval, optvar, active_c = data

    if res_expected == 1:
        assert res == res_expected
        np.testing.assert_allclose(optval, optval_expected)
        np.testing.assert_allclose(optvar, optvar_expected)
        assert set(active_c) == set(active_c_expected)
    else:
        assert res == res_expected

@pytest.mark.parametrize("seed", range(100))
def test_random_constraints(seed):
    """Generate random problem data, solve with cvxpy and then compare!
    Generated problems can be feasible or infeasible. Both cases are
    tested in this unit test.

    """
    # generate random problem data
    d = 50
    np.random.seed(seed)
    seeds = np.random.randint(1000, size=7)
    np.random.seed(seeds[0])
    v = np.random.randn(3)
    np.random.seed(seeds[1])
    a, b = np.random.randn(2, d)
    np.random.seed(seeds[2])
    if seed % 2 == 0:
        c = - np.random.rand(d)
    else:
        c = np.random.randn(d)
    low = np.r_[-0.5, -0.9]
    high = np.r_[0.5, 0.9]
    np.random.seed(seeds[3])
    active_c = np.random.choice(d, size=2)
    # solve with cvxpy
    x = cvx.Variable(2)
    constraints = [a * x[0] + b * x[1] + c <= 0,
                   low <= x, x <= high]
    obj = cvx.Maximize(v[0] * x[0] + v[1] * x[1] + v[2])
    prob = cvx.Problem(obj, constraints)
    prob.solve()
    # solve with the method to test and assert correctness
    data = seidel.solve_lp2d(v, a, b, c, low, high, active_c)
    res, optval, optvar, active_c = data
    if prob.status == "optimal":
        assert res == 1
        np.testing.assert_allclose(optval, prob.value)
        np.testing.assert_allclose(optvar, np.asarray(x.value).flatten())
    elif prob.status == "infeasible":
        assert res == 0
    else:
        assert False, "Solve this LP with cvxpy returns status: {:}".format(prob.status)


def test_err1():
    """A case seidel solver fails to solve correctly. I discovered this
    while working on toppra.

    """
    v = array([-1.e-09,  1.e+00,  0.e+00])
    a = array([-0.02020202,  0.02020202,  1.53515768,  4.3866269 , -3.9954173 , -1.53515768, -4.3866269 ,  3.9954173 ])
    b = array([  -1.        ,    1.        , -185.63664301,  156.27072783, -209.00954213,  185.63664301, -156.27072783,  209.00954213])
    c = array([ 0.       , -0.0062788, -1.       , -2.       , -4.       , -1.       , -1.       , -1.       ])
    low = array([-100.,    0.])
    high = array([1.00000000e+02, 6.26434609e-02])

    data = seidel.solve_lp2d(v, a, b, c, low, high, np.array([0, 5]))  # only break at this active constraints
    res, optval, optvar, active_c = data

    # solve with cvxpy
    x = cvx.Variable(2)
    constraints = [a * x[0] + b * x[1] + c <= 0,
                   low <= x, x <= high]
    obj = cvx.Maximize(v[0] * x[0] + v[1] * x[1] + v[2])
    prob = cvx.Problem(obj, constraints)
    prob.solve()
    # solve with the method to test and assert correctness
    if prob.status == "optimal":
        assert res == 1
        np.testing.assert_allclose(optval, prob.value)
        np.testing.assert_allclose(optvar, np.asarray(x.value).flatten())
    elif prob.status == "infeasible":
        assert res == 0
    else:
        assert False, "Solve this LP with cvxpy returns status: {:}".format(prob.status)


def test_err2():
    """ A case that fails. Discovered on 31/10/2018.
    """
    v=array([-1.e-09,  1.e+00,  0.e+00])
    a=array([-0.04281662,  0.04281662,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        , -1.27049648,  0.63168407,  0.54493736, -0.17238098,
             0.22457236,  0.6543007 ,  1.24159883,  1.27049648, -0.63168407,
             -0.54493736,  0.17238098, -0.22457236, -0.6543007 , -1.24159883])
    b=array([ -1.        ,   1.        , -70.14534325,  35.42759706,
              31.23305996,  -9.04430553,  12.51402852,  36.71562421,
              68.63795557,  70.14534325, -35.42759706, -31.23305996,
              9.04430553, -12.51402852, -36.71562421, -68.63795557,
              -9.70931351,   4.71707751,   3.93518034,  -1.41196299,
              1.69317949,   4.88204872,   9.47085771,   9.70931351,
              -4.71707751,  -3.93518034,   1.41196299,  -1.69317949,
              -4.88204872,  -9.47085771])
    c=array([  0.        ,  -1.56875277, -50.        , -50.        ,
               -50.        , -50.        , -50.        , -50.        ,
               -50.        , -50.        , -50.        , -50.        ,
               -50.        , -50.        , -50.        , -50.        ,
               -50.        , -50.        , -50.        , -50.        ,
               -50.        , -50.        , -50.        , -50.        ,
               -50.        , -50.        , -50.        , -50.        ,
               -50.        , -50.        ])
    low=array([-1.e+08,  0.e+00])
    high=array([1.e+08, 1.e+08])
    active_c=np.array([ 0, -4])

    data = seidel.solve_lp2d(
        v, a, b, c, low, high, np.array([0, -4]))  # only break at this active constraints
    res, optval, optvar, active_c = data

    # solve with cvxpy
    x = cvx.Variable(2)
    constraints = [a * x[0] + b * x[1] + c <= 0,
                   low <= x, x <= high]
    obj = cvx.Maximize(v[0] * x[0] + v[1] * x[1] + v[2])
    prob = cvx.Problem(obj, constraints)
    prob.solve(solver='CVXOPT')
    # solve with the method to test and assert correctness
    if prob.status == "optimal":
        assert res == 1
        np.testing.assert_allclose(optval, prob.value)
        np.testing.assert_allclose(optvar[1], np.asarray(x.value).flatten()[1])
    elif prob.status == "infeasible":
        assert res == 0
    else:
        assert False, "Solve this LP with cvxpy returns status: {:}".format(prob.status)

   
