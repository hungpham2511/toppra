import toppra as ta
import toppra.constraint as constraint
import toppra.algorithm as algo
import numpy as np
import matplotlib.pyplot as plt

ta.setup_logging("INFO")


def main():
    # waypts = [[0], [5], [10]]
    # path = ta.SplineInterpolator([0, 0.5, 1.0], waypts)
    waypts = [[0], [1], [10]]
    path = ta.SplineInterpolator([0, 0.1, 1.0], waypts)
    # NOTE: When constructing a path, you must "align" the waypoint
    # properly yourself. For instance, if the waypoints are [0, 1, 10]
    # like in the above example, the path position should be aligned
    # like [0, 0.1, 1.0]. If this is not done, the CubicSpline
    # Interpolator might result undesirable oscillating paths!
    vlim = np.array([[-3, 3]])
    alim = np.array([[-4, 4]])
    pc_vel = constraint.JointVelocityConstraint(vlim)
    pc_acc = constraint.JointAccelerationConstraint(
        alim, discretization_scheme=constraint.DiscretizationType.Interpolation)

    instance = algo.TOPPRA([pc_vel, pc_acc], path, solver_wrapper='seidel',
                           gridpoints=np.linspace(0, 1.0, 1001))
    jnt_traj, aux_traj = instance.compute_trajectory(0, 0)

    duration = jnt_traj.duration
    print("Found optimal trajectory with duration {:f} sec".format(duration))
    ts = np.linspace(0, duration, 100)
    fig, axs = plt.subplots(3, 1, sharex=True)
    qs = jnt_traj.eval(ts)
    qds = jnt_traj.evald(ts)
    qdds = jnt_traj.evaldd(ts)
    axs[0].plot(ts, qs)
    axs[1].plot(ts, qds)
    axs[2].plot(ts, qdds)
    plt.show()

    # NOTE: uncomment the below lines to drop into IPython for
    # inspection. Note that you need to have IPython installed.

    # import IPython
    # if IPython.get_ipython() is None:
        # IPython.embed()

if __name__ == '__main__':
    main()
