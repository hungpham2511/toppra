import pytest
import numpy as np
import toppra


@pytest.fixture
def coefficients_functions():
    "Coefficients for the equation: A(q) ddot q + dot q B(q) dot q + C(q) = w"
    def A(q):
        return np.array([[np.sin(q[0]), 0],
                         [np.cos(q[1]), q[0] + q[1]]])

    def B(q):
        ret = np.zeros((2, 2, 2))
        ret[:, :, 0] = [[np.sin(2 * q[0]), 0], [0, q[1] ** 2]]
        ret[:, :, 1] = [[1, 2], [3, q[0]]]
        return ret

    def C(q):
        return np.array([q[0] * q[1], 0])

    def F(q):
        ret = np.zeros((4, 2))
        ret[:, 0] = [1, 1, 1, 1]
        ret[:, 1] = [10 * np.sin(q[0]), q[1], q[0] + q[1], 10]
        return ret

    def g(q):
        return np.array([100, 200])

    np.random.seed(0)
    path = toppra.SplineInterpolator([0, 1, 2, 4], np.random.randn(4, 2))
    return A, B, C, F, g, path


def test_wrong_dimension(coefficients_functions):
    A, B, C, cnst_F, cnst_g, path = coefficients_functions
    def inv_dyn(q, qd, qdd):
        return A(q).dot(qdd) + np.dot(qd.T, np.dot(B(q), qd)) + C(q)
    constraint = toppra.constraint.CanonicalLinearSecondOrderConstraint(inv_dyn, cnst_F, cnst_g, dof=2)
    path_wrongdim = toppra.SplineInterpolator(np.linspace(0, 1, 5), np.random.randn(5, 10))
    with pytest.raises(AssertionError) as e_info:
        constraint.compute_constraint_params(path_wrongdim, np.r_[0, 0.5, 1], 1.0)
    assert e_info.value.args[0] == "Wrong dimension: constraint dof ({:d}) not equal to path dof ({:d})".format(
        constraint.get_dof(), 10
    )


def test_assemble_ABCFg(coefficients_functions):
    """ For randomly set A, B, C, F, g functions. The generated parameters must equal
    those given by equations.
    """
    A, B, C, cnst_F, cnst_g, path = coefficients_functions

    def inv_dyn(q, qd, qdd):
        return A(q).dot(qdd) + np.dot(qd.T, np.dot(B(q), qd)) + C(q)
    constraint = toppra.constraint.CanonicalLinearSecondOrderConstraint(inv_dyn, cnst_F, cnst_g, dof=2)
    constraint.set_discretization_type(0)
    a, b, c, F, g, _, _ = constraint.compute_constraint_params(
        path, np.linspace(0, path.get_duration(), 10), 1.0)

    # Correct params
    q_vec = path.eval(np.linspace(0, path.get_duration(), 10))
    qs_vec = path.evald(np.linspace(0, path.get_duration(), 10))
    qss_vec = path.evaldd(np.linspace(0, path.get_duration(), 10))

    for i in range(10):
        ai_ = A(q_vec[i]).dot(qs_vec[i])
        bi_ = A(q_vec[i]).dot(qss_vec[i]) + np.dot(qs_vec[i].T, B(q_vec[i]).dot(qs_vec[i]))
        ci_ = C(q_vec[i])
        np.testing.assert_allclose(ai_, a[i])
        np.testing.assert_allclose(bi_, b[i])
        np.testing.assert_allclose(ci_, c[i])
        np.testing.assert_allclose(cnst_F(q_vec[i]), F[i])
        np.testing.assert_allclose(cnst_g(q_vec[i]), g[i])
