import toppra as ta
import toppra.constraint as constraint
import toppra.algorithm as algo
import numpy as np
import matplotlib.pyplot as plt
import time

ta.setup_logging("INFO")


def generate_new_problem(seed=9):
    # Parameters
    N_samples = 5
    dof = 7
    np.random.seed(seed)
    way_pts = np.random.randn(N_samples, dof)
    return (
        np.linspace(0, 1, 5),
        way_pts,
        10 + np.random.rand(dof) * 20,
        10 + np.random.rand(dof) * 2,
    )


def main():
    ss, way_pts, vlims, alims = generate_new_problem()
    path = ta.SplineInterpolator(ss, way_pts)
    pc_vel = constraint.JointVelocityConstraint(vlims)
    pc_acc = constraint.JointAccelerationConstraint(alims)

    # Setup a parametrization instance. The keyword arguments are optional.
    instance = algo.TOPPRA([pc_vel, pc_acc], path)

    # Retime the trajectory, only this step is necessary.
    t0 = time.time()
    jnt_traj = instance.compute_trajectory()

    # return_data flag outputs internal data obtained while computing
    # the paramterization. This include the time stamps corresponding
    # to the original waypoints. See below (line 53) to see how to
    # extract the time stamps.
    print("Parameterization time: {:} secs".format(time.time() - t0))

    instance.compute_feasible_sets()

    ts_sample = np.linspace(0, jnt_traj.duration, 100)
    qs_sample = jnt_traj(ts_sample)
    for i in range(path.dof):
        # plot the i-th joint trajectory
        plt.plot(ts_sample, qs_sample[:, i], c="C{:d}".format(i))
    plt.xlabel("Time (s)")
    plt.ylabel("Joint position (rad/s^2)")
    plt.show()


    K = instance.problem_data.K
    X = instance.problem_data.X

    plt.plot(X[:, 0], c="green", label="Feasible sets")
    plt.plot(X[:, 1], c="green")
    plt.plot(K[:, 0], "--", c="red", label="Controllable sets")
    plt.plot(K[:, 1], "--", c="red")
    plt.plot(instance.problem_data.sd_vec ** 2, label="Velocity profile")
    plt.title("Path-position path-velocity plot")
    plt.xlabel("Path position")
    plt.ylabel("Path velocity square")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
