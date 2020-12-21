import toppra
import toppra.interpolator
import numpy as np
import matplotlib.pyplot as plt
import pytest


@pytest.fixture(params=[[0, 1], [1.5, 2.7]])
def path(request):
    start, end = request.param
    waypoints = [[0, 0.3, 0.5], [1, 2, 3], [0.0, 0.1, 0.2], [0, 0.5, 0]]
    ss = np.linspace(start, end, len(waypoints))
    path = toppra.interpolator.SplineInterpolator(ss, waypoints)
    yield path, waypoints


def test_basic_usage(path):
    path, waypoints = path
    gridpoints_ept = toppra.interpolator.propose_gridpoints(path, 1e-2)
    assert gridpoints_ept[0] == path.path_interval[0]
    assert gridpoints_ept[-1] == path.path_interval[1]

    # The longest segment should be smaller than 0.1. This is to
    # ensure a reasonable response.
    assert np.max(np.diff(gridpoints_ept)) < 0.05

    # # visualize ###############################################################
    # ss_full = np.linspace(path.path_interval[0], path.path_interval[1], 100)
    # for i in range(len(waypoints[0])):
    #     plt.plot(ss_full, path(ss_full)[:, i], '--', c='C%d' % i)
    #     plt.plot(gridpoints_ept, path(gridpoints_ept)[:, i], '-o', c='C%d' % i)
    # plt.show()


def test_number_of_points_(path):
     path, waypoints = path
     gridpoints_ept = toppra.interpolator.propose_gridpoints(path, 1.0, min_nb_points=100)  # large bounds
     assert len(gridpoints_ept) > 100


def test_hard_path_difficult_to_approximate_within_iterations(path):
    """The given setting makes the approximation fails."""
    path, _ = path
    with pytest.raises(ValueError):
        toppra.interpolator.propose_gridpoints(path, max_iteration=2)
