import pytest
import numpy as np
import numpy.testing as npt
import toppra as ta
import toppra.constraint as constraint
from toppra.constants import JACC_MAXU


@pytest.fixture(params=[1, 2, 6, '6d'], name='accel_constraint_setup')
def create_acceleration_pc_fixtures(request):
    """ Parameterized Acceleration path constraint.

    Return:
    -------
      data: A tuple. Contains path, ss, alim.
      pc: A `PathConstraint`.
    """
    dof = request.param
    if dof == 1:  # Scalar
        pi = ta.PolynomialPath([1, 2, 3])  # 1 + 2s + 3s^2
        ss = np.linspace(0, 1, 3)
        alim = (np.r_[-1., 1]).reshape(1, 2)  # Scalar case
        accel_const = constraint.JointAccelerationConstraint(alim, constraint.DiscretizationType.Collocation)
        data = (pi, ss, alim)
        return data, accel_const

    if dof == 2:
        coeff = [[1., 2, 3], [-2., -3., 4., 5.]]
        pi = ta.PolynomialPath(coeff)
        ss = np.linspace(0, 0.75, 4)
        alim = np.array([[-1., 2], [-2., 2]])
        accel_const = constraint.JointAccelerationConstraint(alim, constraint.DiscretizationType.Collocation)
        data = (pi, ss, alim)
        return data, accel_const

    if dof == 6:
        np.random.seed(10)
        N = 20
        way_pts = np.random.randn(10, 6)
        pi = ta.SplineInterpolator(np.linspace(0, 1, 10), way_pts)
        ss = np.linspace(0, 1, N + 1)
        vlim_ = np.random.rand(6)
        alim = np.vstack((-vlim_, vlim_)).T
        accel_const = constraint.JointAccelerationConstraint(alim, constraint.DiscretizationType.Collocation)
        data = (pi, ss, alim)
        return data, accel_const

    if dof == '6d':
        np.random.seed(10)
        N = 20
        way_pts = np.random.randn(10, 6)
        pi = ta.SplineInterpolator(np.linspace(0, 1, 10), way_pts)
        ss = np.linspace(0, 1, N + 1)
        alim_s = np.random.rand(6)
        alim = np.vstack((-alim_s, alim_s)).T
        accel_const = constraint.JointAccelerationConstraint(alim_s, constraint.DiscretizationType.Collocation)
        data = (pi, ss, alim)
        return data, accel_const


def test_constraint_type(accel_constraint_setup):
    """ Syntactic correct.
    """
    data, pc = accel_constraint_setup
    assert pc.get_constraint_type() == constraint.ConstraintType.CanonicalLinear


def test_constraint_params(accel_constraint_setup):
    """ Test constraint satisfaction with cvxpy.
    """
    (path, ss, alim), accel_const = accel_constraint_setup

    # An user of the class
    a, b, c, F, g, ubound, xbound = accel_const.compute_constraint_params(path, ss)
    assert xbound is None

    N = ss.shape[0] - 1
    dof = path.dof

    ps = path.evald(ss)
    pss = path.evaldd(ss)

    F_actual = np.vstack((np.eye(dof), - np.eye(dof)))
    g_actual = np.hstack((alim[:, 1], - alim[:, 0]))

    npt.assert_allclose(F, F_actual)
    npt.assert_allclose(g, g_actual)
    for i in range(0, N + 1):
        npt.assert_allclose(a[i], ps[i])
        npt.assert_allclose(b[i], pss[i])
        npt.assert_allclose(c[i], np.zeros_like(ps[i]))
        assert ubound is None
        assert xbound is None


def test_wrong_dimension(accel_constraint_setup):
    _, path_constraint = accel_constraint_setup
    path_wrongdim = ta.SplineInterpolator(np.linspace(0, 1, 5), np.random.randn(5, 10))
    with pytest.raises(ValueError) as e_info:
        path_constraint.compute_constraint_params(path_wrongdim, np.r_[0, 0.5, 1], 1.0)
    assert e_info.value.args[0] == "Wrong dimension: constraint dof ({:d}) not equal to path dof ({:d})".format(
        path_constraint.get_dof(), 10
    )


