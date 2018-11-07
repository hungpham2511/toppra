"""This test suite contains tests to verify that TOPPRA can produce
reasonable results to trajectories of very small movements. Almost
zero.
"""
import pytest
import numpy as np
import toppra

toppra.setup_logging(level="INFO")


@pytest.mark.parametrize("scaling", [1e-1, 1e-2, 1e-3])
@pytest.mark.parametrize("Ngrid", [101, 501, 1001])
def test_scalar_zero_motion(scaling, Ngrid):
    """ The simple zero motion trajectory
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
        [pc_vel, pc_acc], path, solver_wrapper='seidel',
        gridpoints=np.linspace(0, 1.0, Ngrid), scaling=scaling)
    jnt_traj, aux_traj, data = instance.compute_trajectory(0, 0, return_data=True)
    # Simply assert success
    assert jnt_traj is not None

