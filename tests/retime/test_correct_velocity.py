import pytest
import numpy as np
import toppra
import toppra.constraint as constraint


@pytest.fixture(params=["spline", "poly"])
def basic_path(request):
    """ Return a generic path.
    """
    if request.param == "spline":
        np.random.seed(1)
        path = toppra.SplineInterpolator(np.linspace(0, 1, 5), np.random.randn(5, 7))
    elif request.param == "poly":
        np.random.seed(1)
        coeffs = np.random.randn(7, 3)  # 7 random quadratic equations
        path = toppra.PolynomialPath(coeffs)
    yield path


@pytest.mark.parametrize("solver_wrapper", ["hotqpoases", "seidel"])
def test_zero_velocity(basic_constraints, basic_path, solver_wrapper):
    """Check that initial and final velocity are correct."""
    vel_c, acc_c, ro_acc_c = basic_constraints
    instance = toppra.algorithm.TOPPRA([vel_c, acc_c], basic_path,
                                       solver_wrapper=solver_wrapper)
    jnt_traj, _ = instance.compute_trajectory(0, 0)

    # assertion
    initial_velocity = jnt_traj(0, 1)
    final_velocity = jnt_traj(jnt_traj.duration, 1)
    np.testing.assert_allclose(initial_velocity, 0, atol=1e-7)
    np.testing.assert_allclose(final_velocity, 0, atol=1e-7)


@pytest.mark.parametrize("velocity_start", [0, 0.1])
@pytest.mark.parametrize("velocity_end", [0, 0.1])
@pytest.mark.parametrize("solver_wrapper", ["hotqpoases", "seidel"])
def test_nonzero_velocity(velocity_start, velocity_end, basic_constraints, basic_path, solver_wrapper):
    vel_c, acc_c, ro_acc_c = basic_constraints
    instance = toppra.algorithm.TOPPRA([vel_c, acc_c], basic_path,
                                       solver_wrapper=solver_wrapper)
    jnt_traj, _ = instance.compute_trajectory(velocity_start, velocity_end)

    # assertion
    initial_velocity = jnt_traj(0, 1)
    initial_velocity_expt = basic_path(0, 1) * velocity_start

    final_velocity = jnt_traj(jnt_traj.duration, 1)
    final_velocity_expt = basic_path(basic_path.duration, 1) * velocity_end
    np.testing.assert_allclose(initial_velocity, initial_velocity_expt, atol=1e-7)
    np.testing.assert_allclose(final_velocity, final_velocity_expt, atol=1e-7)


@pytest.mark.parametrize("solver_wrapper", ["hotqpoases", "seidel"])
@pytest.mark.parametrize("velocities", [[0, -1], [-1, 0], [-1, -1]])
def test_invalid_velocity(velocities, basic_constraints, basic_path, solver_wrapper):
    vel_c, acc_c, ro_acc_c = basic_constraints
    instance = toppra.algorithm.TOPPRA([vel_c, acc_c], basic_path,
                                       solver_wrapper=solver_wrapper)
    with pytest.raises(ValueError) as err:
        jnt_traj, _ = instance.compute_trajectory(*velocities)
    assert "Negative" in err.value.args[0]

