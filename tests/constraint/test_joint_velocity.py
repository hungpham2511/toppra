import pytest
import cvxpy as cvx
import numpy as np
import numpy.testing as npt
import toppra as ta
import toppra.constraint as constraint
from toppra.constants import TINY


@pytest.fixture(scope="class", params=[2, 6], name='velocity_pc_data')
def create_velocity_pc_fixtures(request):
    """Parameterized fixture to test Velocity constraint.

    Return:
    -------
      data: A tuple. Contains path, ss, vim.
      pc: A `PathConstraint`.

    """
    if request.param == 2:
        coeff = [[1., 2, 3], [-2., -3., 4., 5.]]
        pi = ta.PolynomialPath(coeff)
        ss = np.linspace(0, 0.75, 4)
        vlim = np.array([[-1., 2], [-2., 2]])
        velocity_constraint = constraint.JointVelocityConstraint(vlim)
        data = (pi, ss, vlim)
        return data, velocity_constraint

    if request.param == 6:
        np.random.seed(10)
        N = 100
        way_pts = np.random.randn(10, 6)
        pi = ta.SplineInterpolator(np.linspace(0, 1, 10), way_pts)
        ss = np.linspace(0, 1, N + 1)
        vlim_ = np.random.rand(6) * 10 + 2.
        vlim = np.vstack((-vlim_, vlim_)).T
        vel_constraint = constraint.JointVelocityConstraint(vlim)
        data = (pi, ss, vlim)
        return data, vel_constraint


class TestClass_JointVelocityConstraint(object):
    def test_constraint_type(self, velocity_pc_data):
        """ Syntactic correctness: the object returned should have correct dimension.
        """
        data, pc = velocity_pc_data
        assert pc.get_constraint_type() == constraint.ConstraintType.CanonicalLinear

    def test_constraint_satisfaction(self, velocity_pc_data):
        """ Test constraint satisfaction with cvxpy.
        """
        data, pc = velocity_pc_data
        path, ss, vlim = data

        constraint_param = pc.compute_constraint_params(path, ss)
        _, _, _, _, _, _, xlimit = constraint_param

        qs = path.evald(ss)
        N = ss.shape[0] - 1

        sd = cvx.Variable()

        for i in range(0, N + 1):
            # 2. Compute max sd from the data
            constraints = [qs[i] * sd <= vlim[:, 1],
                           qs[i] * sd >= vlim[:, 0],
                           sd >= 0]
            prob = cvx.Problem(cvx.Maximize(sd), constraints)
            prob.solve(solver=cvx.ECOS, abstol=1e-9)
            xmax = sd.value ** 2

            prob = cvx.Problem(cvx.Minimize(sd), constraints)
            prob.solve(solver=cvx.ECOS, abstol=1e-9)
            xmin = sd.value ** 2

            # 3. They should agree
            npt.assert_allclose([xmin, xmax], xlimit[i], atol=TINY)

            # Assert non-negativity
            assert xlimit[i, 0] >= 0

    def test_wrong_dimension(self, velocity_pc_data):
        data, pc = velocity_pc_data
        path_wrongdim = ta.SplineInterpolator(np.linspace(0, 1, 5), np.random.randn(5, 10))
        with pytest.raises(ValueError) as e_info:
            pc.compute_constraint_params(path_wrongdim, [0, 0.5, 1])
        assert e_info.value.args[0] == "Wrong dimension: constraint dof ({:d}) not equal to path dof ({:d})".format(
            pc.get_dof(), 10
        )


