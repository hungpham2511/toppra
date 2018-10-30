import pytest
import numpy as np
import numpy.testing as npt
import toppra as ta
import toppra.constraint as constraint
from toppra.constants import JACC_MAXU


@pytest.fixture(scope="class", params=[1, 2, 6], name='acceleration_pc_data')
def create_acceleration_pc_fixtures(request):
    """ Parameterized Acceleration path constraint.

    Return:
    -------
      data: A tuple. Contains path, ss, alim.
      pc: A `PathConstraint`.
    """
    if request.param == 1:  # Scalar
        pi = ta.PolynomialPath([1, 2, 3])  # 1 + 2s + 3s^2
        ss = np.linspace(0, 1, 3)
        alim = (np.r_[-1., 1]).reshape(1, 2)  # Scalar case
        pc_vel = constraint.JointAccelerationConstraint(alim)
        data = (pi, ss, alim)
        return data, pc_vel

    if request.param == 2:
        coeff = [[1., 2, 3], [-2., -3., 4., 5.]]
        pi = ta.PolynomialPath(coeff)
        ss = np.linspace(0, 0.75, 4)
        alim = np.array([[-1., 2], [-2., 2]])
        pc_vel = constraint.JointAccelerationConstraint(alim)
        data = (pi, ss, alim)
        return data, pc_vel

    if request.param == 6:
        np.random.seed(10)
        N = 20
        way_pts = np.random.randn(10, 6)
        pi = ta.SplineInterpolator(np.linspace(0, 1, 10), way_pts)
        ss = np.linspace(0, 1, N + 1)
        vlim_ = np.random.rand(6)
        alim = np.vstack((-vlim_, vlim_)).T
        pc_vel = constraint.JointAccelerationConstraint(alim)
        data = (pi, ss, alim)
        return data, pc_vel


class TestClass_JointAccelerationConstraint(object):
    """

    Tests:
    ------

    1. syntactic: the object return should have correct dimension.

    2. constraint satisfaction: the `PathConstraint` returned should
    be consistent with the data.

    """
    def test_constraint_type(self, acceleration_pc_data):
        """ Syntactic correct.
        """
        data, pc = acceleration_pc_data
        assert pc.get_constraint_type() == constraint.ConstraintType.CanonicalLinear

    def test_constraint_params(self, acceleration_pc_data):
        """ Test constraint satisfaction with cvxpy.
        """
        data, constraint = acceleration_pc_data
        path, ss, alim = data

        # An user of the class
        a, b, c, F, g, ubound, xbound = constraint.compute_constraint_params(path, ss, 1.0)
        assert xbound is None

        N = ss.shape[0] - 1
        dof = path.get_dof()

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
            npt.assert_allclose(ubound[i], [-JACC_MAXU, JACC_MAXU])

    def test_wrong_dimension(self, acceleration_pc_data):
        data, pc = acceleration_pc_data
        path_wrongdim = ta.SplineInterpolator(np.linspace(0, 1, 5), np.random.randn(5, 10))
        with pytest.raises(ValueError) as e_info:
            pc.compute_constraint_params(path_wrongdim, np.r_[0, 0.5, 1], 1.0)
        assert e_info.value.args[0] == "Wrong dimension: constraint dof ({:d}) not equal to path dof ({:d})".format(
            pc.get_dof(), 10
        )


