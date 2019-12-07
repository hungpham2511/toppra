import toppra
import toppra.interpolator
import numpy as np
import matplotlib.pyplot as plt


def test_basic_usage():
    waypoints = [[0, 0.3, 0.5], [1, 2, 3], [0., 0.1, 0.2], [0, 0.5, 0]]
    ss = np.linspace(0, 1, len(waypoints))
    path = toppra.interpolator.SplineInterpolator(ss, waypoints)

    gridpoints_ept = toppra.interpolator.propose_gridpoints(path)

    # visualize ###############################################################
    ss_full = np.linspace(0, 1, 100)
    for i in range(len(waypoints[0])):
        plt.plot(ss_full, path(ss_full)[:, i], '--', c='C%d' % i)
        plt.plot(gridpoints_ept, path(gridpoints_ept)[:, i], '-o', c='C%d' % i)
    plt.show()
    
    
