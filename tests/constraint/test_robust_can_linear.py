import toppra
import pytest
import numpy as np


@pytest.fixture()
def accel_constraint(request):
    dof = 5
    np.random.seed(0)
    alim_ = np.random.rand(5)
    alim = np.vstack((-alim_, alim_)).T
    constraint = toppra.constraint.JointAccelerationConstraint(alim)

    np.random.seed(0)
    path = toppra.SplineInterpolator(np.linspace(0, 1, 5), np.random.randn(5, dof))
    yield constraint, path

@pytest.mark.parametrize("dist_scheme", [toppra.constraint.DiscretizationType.Collocation,
                                         toppra.constraint.DiscretizationType.Interpolation,
                                         0, 1])
def test_basic(accel_constraint, dist_scheme):
    "Basic initialization."
    cnst, path = accel_constraint

    ro_cnst = toppra.constraint.RobustCanonicalLinearConstraint(cnst, [0.1, 2, .3], dist_scheme)

    assert ro_cnst.get_constraint_type() == toppra.constraint.ConstraintType.CanonicalConic
    assert ro_cnst.get_dof() == 5

    a, b, c, P = ro_cnst.compute_constraint_params(
        path, np.linspace(0, path.get_duration(), 10), 1.0)
    d = a.shape[1] - 2

    # assert a.shape == (10, 2 * path.get_dof())
    # assert b.shape == (10, 2 * path.get_dof())
    # assert c.shape == (10, 2 * path.get_dof())
    # assert P.shape == (10, 2 * path.get_dof(), 3, 3)

    # Linear params
    cnst.set_discretization_type(dist_scheme)
    a0, b0, c0, F0, g0, _, _ = cnst.compute_constraint_params(
        path, np.linspace(0, path.get_duration(), 10), 1.0)

    # Assert values
    for i in range(10):
        np.testing.assert_allclose(a[i, :d], F0.dot(a0[i]))
        np.testing.assert_allclose(b[i, :d], F0.dot(b0[i]))
        np.testing.assert_allclose(c[i, :d], F0.dot(c0[i]) - g0)
    for i in range(10):
        for j in range(a0.shape[1]):
            np.testing.assert_allclose(P[i, j], np.diag([0.1, 2, .3]))


def test_negative_perb(accel_constraint):
    "If negative pertubations are given, raise ValueError"
    cnst, path = accel_constraint
    with pytest.raises(ValueError) as e_info:
        ro_cnst = toppra.constraint.RobustCanonicalLinearConstraint(cnst, [-0.1, 2, .3])
    assert e_info.value.args[0] == "Perturbation must be non-negative. Input {:}".format([-0.1, 2, .3])


