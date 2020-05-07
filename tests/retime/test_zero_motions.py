"""This test suite contains tests to verify that TOPPRA can produce
reasonable results to trajectories of very small movements. Almost
zero.
"""
import pytest
import numpy as np
import toppra

toppra.setup_logging(level="INFO")

# TODO:It's better to simple return failure whenever zero motion is
# detected. This will belongs to the case of ERR_NUMERICAL_DIFFICULTY

@pytest.mark.skip(reason="Do not handle zero motion case.")
@pytest.mark.parametrize("Ngrid", [101, 1001])
def test_scalar_zero_motion(Ngrid):
    """The simple zero motion trajectory

    Note: Paths with very small displacement, like the one given in
    this example, are pathological: the optimal path velocities and
    accelerations tend to be extremely large and have orders of
    magnitudes difference. Because of this reason, seidel solver
    wrapper is unlikely to work well and therefore is not tested.

    """
    waypts = [[0], [1e-8], [0]]
    path = toppra.SplineInterpolator([0, 0.5, 1.0], waypts)
    # NOTE: When constructing a path, you must "align" the waypoint
    # properly yourself. For instance, if the waypoints are [0, 1, 10]
    # like in the above example, the path position should be aligned
    # like [0, 0.1, 1.0]. If this is not done, the CubicSpline
    # Interpolator might result undesirable oscillating paths!
    vlim = np.array([[-10, 10]])
    alim = np.array([[-4, 4]])
    pc_vel = toppra.constraint.JointVelocityConstraint(vlim)
    pc_acc = toppra.constraint.JointAccelerationConstraint(
        alim, discretization_scheme=toppra.constraint.DiscretizationType.Interpolation)

    instance = toppra.algorithm.TOPPRA(
        [pc_vel, pc_acc], path, solver_wrapper='hotqpoases',
        gridpoints=np.linspace(0, 1.0, Ngrid))
    jnt_traj = instance.compute_trajectory(0, 0, return_data=True)
    # Simply assert success
    assert jnt_traj is not None
    assert jnt_traj.duration < 9e-4  # less than 1ms


@pytest.mark.skip(reason="Do not handle zero motion case.")
@pytest.mark.parametrize("Ngrid", [101, 501, 1001])
def test_scalar_auto_scaling(Ngrid):
    """Automatic scaling should lead to better results at slower
    trajectories.
    """
    waypts = [[0], [1e-8], [0]]
    path = toppra.SplineInterpolator([0, 0.5, 1.0], waypts)
    # NOTE: When constructing a path, you must "align" the waypoint
    # properly yourself. For instance, if the waypoints are [0, 1, 10]
    # like in the above example, the path position should be aligned
    # like [0, 0.1, 1.0]. If this is not done, the CubicSpline
    # Interpolator might result undesirable oscillating paths!
    vlim = np.array([[-10, 10]])
    alim = np.array([[-4, 4]])
    pc_vel = toppra.constraint.JointVelocityConstraint(vlim)
    pc_acc = toppra.constraint.JointAccelerationConstraint(
        alim, discretization_scheme=toppra.constraint.DiscretizationType.Interpolation)

    instance = toppra.algorithm.TOPPRA(
        [pc_vel, pc_acc], path, solver_wrapper='hotqpoases',
        gridpoints=np.linspace(0, 1.0, Ngrid))
    jnt_traj = instance.compute_trajectory(0, 0, return_data=True)

    # Simply assert success
    assert jnt_traj is not None
    assert jnt_traj.duration < 9e-4  # less than 1 ms

