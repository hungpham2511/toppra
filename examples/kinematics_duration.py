import toppra as ta
import toppra.constraint as constraint
import toppra.algorithm as algo
import numpy as np
import matplotlib.pyplot as plt
import time


ta.setup_logging("INFO")


def main():
    # Parameters
    N_samples = 5
    SEED = 9
    dof = 7

    # Random waypoints used to obtain a random geometric path. Here,
    # we use spline interpolation.
    np.random.seed(SEED)
    way_pts = np.random.randn(N_samples, dof)
    path = ta.SplineInterpolator(np.linspace(0, 1, 5), way_pts)

    # Create velocity bounds, then velocity constraint object
    vlim_ = np.random.rand(dof) * 20
    vlim = np.vstack((-vlim_, vlim_)).T
    # Create acceleration bounds, then acceleration constraint object
    alim_ = np.random.rand(dof) * 2
    alim = np.vstack((-alim_, alim_)).T
    pc_vel = constraint.JointVelocityConstraint(vlim)
    pc_acc = constraint.JointAccelerationConstraint(
        alim, discretization_scheme=constraint.DiscretizationType.Interpolation)

    # Setup a parametrization instance
    instance = algo.TOPPRAsd([pc_vel, pc_acc], path, gridpoints=np.linspace(0, 1, 101),
                             solver_wrapper='seidel')
    instance.set_desired_duration(60)
    t0 = time.time()
    # Retime the trajectory, only this step is necessary.
    jnt_traj, aux_traj = instance.compute_trajectory(0, 0)
    print("Parameterization took {:} secs".format(time.time() - t0))
    ts_sample = np.linspace(0, jnt_traj.get_duration(), 100)
    qs_sample = jnt_traj.evaldd(ts_sample)

    plt.plot(ts_sample, qs_sample)
    plt.xlabel("Time (s)")
    plt.ylabel("Joint acceleration (rad/s^2)")
    plt.show()

    # Compute the feasible sets and the controllable sets for viewing.
    # Note that these steps are not necessary.
    X = instance.compute_feasible_sets()
    K = instance.compute_controllable_sets(0, 0)

    _, sd_vec, _ = instance.compute_parameterization(0, 0)

    X = np.sqrt(X)
    K = np.sqrt(K)

    plt.plot(X[:, 0], c='green', label="Feasible sets")
    plt.plot(X[:, 1], c='green')
    plt.plot(K[:, 0], '--', c='red', label="Controllable sets")
    plt.plot(K[:, 1], '--', c='red')
    plt.plot(sd_vec, label="Velocity profile")
    plt.title("Path-position path-velocity plot")
    plt.xlabel("Path position")
    plt.ylabel("Path velocity square")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
