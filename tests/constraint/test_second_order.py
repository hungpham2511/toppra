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

    def inv_dyn(q, qd, qdd):
        return A(q).dot(qdd) + np.dot(qd.T, np.dot(B(q), qd)) + C(q)

    np.random.seed(0)
    path = toppra.SplineInterpolator([0, 1, 2, 4], np.random.randn(4, 2))
    return A, B, C, F, g, path, inv_dyn


def test_wrong_dimension(coefficients_functions):
    """If the given path has wrong dimension, raise error."""
    A, B, C, cnst_F, cnst_g, path, inv_dyn = coefficients_functions
    constraint = toppra.constraint.SecondOrderConstraint(inv_dyn, cnst_F, cnst_g, dof=2)
    path_wrongdim = toppra.SplineInterpolator(np.linspace(0, 1, 5), np.random.randn(5, 10))
    with pytest.raises(ValueError) as e_info:
        constraint.compute_constraint_params(path_wrongdim, np.r_[0, 0.5, 1], 1.0)
    assert e_info.value.args[0] == "Wrong dimension: constraint dof ({:d}) not equal to path dof ({:d})".format(
        constraint.dof, 10)


def test_correctness(coefficients_functions):
    """ For randomly set A, B, C, F, g functions. The generated parameters must equal
    those given by equations.
    """
    A, B, C, cnst_F, cnst_g, path, inv_dyn = coefficients_functions
    constraint = toppra.constraint.SecondOrderConstraint(
        inv_dyn, cnst_F, cnst_g, dof=2,
        discretization_scheme=toppra.constraint.DiscretizationType.Collocation)
    a, b, c, F, g, _, _ = constraint.compute_constraint_params(
        path, np.linspace(0, path.duration, 10), 1.0)

    # Correct params
    q_vec = path.eval(np.linspace(0, path.duration, 10))
    qs_vec = path.evald(np.linspace(0, path.duration, 10))
    qss_vec = path.evaldd(np.linspace(0, path.duration, 10))

    for i in range(10):
        ai_ = A(q_vec[i]).dot(qs_vec[i])
        bi_ = A(q_vec[i]).dot(qss_vec[i]) + np.dot(qs_vec[i].T, B(q_vec[i]).dot(qs_vec[i]))
        ci_ = C(q_vec[i])
        np.testing.assert_allclose(ai_, a[i])
        np.testing.assert_allclose(bi_, b[i])
        np.testing.assert_allclose(ci_, c[i])
        np.testing.assert_allclose(cnst_F(q_vec[i]), F[i])
        np.testing.assert_allclose(cnst_g(q_vec[i]), g[i])


@pytest.fixture
def friction():
    def randomized_friction(q):
        """Randomize with fixed input/output."""
        np.random.seed(int(abs(np.sum(q)) * 1000))
        return 2 + np.sin(q) + np.random.rand(len(q))
    yield randomized_friction


def test_joint_torque(coefficients_functions, friction):
    """ Same as the above test, but has frictional effect.
    """
    # setup
    A, B, C, cnst_F, cnst_g, path, _ = coefficients_functions
    def inv_dyn(q, qd, qdd):
        return A(q).dot(qdd) + np.dot(qd.T, np.dot(B(q), qd)) + C(q)
    friction = np.random.rand(2)
    taulim = np.random.randn(2, 2)
    constraint = toppra.constraint.SecondOrderConstraint.joint_torque_constraint(
        inv_dyn, taulim, friction)
    constraint.set_discretization_type(0)
    a, b, c, F, g, _, _ = constraint.compute_constraint_params(
        path, np.linspace(0, path.duration, 10), 1.0)

    # Correct params
    p_vec = path.eval(np.linspace(0, path.duration, 10))
    ps_vec = path.evald(np.linspace(0, path.duration, 10))
    pss_vec = path.evaldd(np.linspace(0, path.duration, 10))

    dof = 2
    F_actual = np.vstack((np.eye(dof), - np.eye(dof)))
    g_actual = np.hstack((taulim[:, 1], - taulim[:, 0]))

    for i in range(10):
        ai_ = A(p_vec[i]).dot(ps_vec[i])
        bi_ = A(p_vec[i]).dot(pss_vec[i]) + np.dot(ps_vec[i].T, B(p_vec[i]).dot(ps_vec[i]))
        ci_ = C(p_vec[i]) + np.sign(ps_vec[i]) * friction
        np.testing.assert_allclose(ai_, a[i])
        np.testing.assert_allclose(bi_, b[i])
        np.testing.assert_allclose(ci_, c[i])
        np.testing.assert_allclose(F_actual, F[i])
        np.testing.assert_allclose(g_actual, g[i])
