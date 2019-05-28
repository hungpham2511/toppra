import toppra as ta
import toppra.constraint as constraint
import toppra.algorithm as algo
import numpy as np
import matplotlib.pyplot as plt
import time
import openravepy as orpy

ta.setup_logging("INFO")


def main():
    # openrave setup
    env = orpy.Environment()
    env.Load("robots/barrettwam.robot.xml")
    env.SetViewer('qtosg')
    robot = env.GetRobots()[0]

    robot.SetActiveDOFs(range(7))

    # Parameters
    N_samples = 5
    SEED = 9
    dof = 7

    # Random waypoints used to obtain a random geometric path. Here,
    # we use spline interpolation.
    np.random.seed(SEED)
    way_pts = np.random.randn(N_samples, dof) * 0.6
    path = ta.SplineInterpolator(np.linspace(0, 1, 5), way_pts)

    # Create velocity bounds, then velocity constraint object
    vlim_ = robot.GetActiveDOFMaxVel()
    vlim_[robot.GetActiveDOFIndices()] = [80., 80., 80., 80., 80., 80., 80.]
    vlim = np.vstack((-vlim_, vlim_)).T
    # Create acceleration bounds, then acceleration constraint object
    alim_ = robot.GetActiveDOFMaxAccel()
    alim_[robot.GetActiveDOFIndices()] = [80., 80., 80., 80., 80., 80., 80.]
    alim = np.vstack((-alim_, alim_)).T
    pc_vel = constraint.JointVelocityConstraint(vlim)
    pc_acc = constraint.JointAccelerationConstraint(
        alim, discretization_scheme=constraint.DiscretizationType.Interpolation)

    # torque limit
    def inv_dyn(q, qd, qdd):
        qdd_full = np.zeros(robot.GetDOF())
        active_dofs = robot.GetActiveDOFIndices()
        with robot:
            # Temporary remove vel/acc constraints
            vlim = robot.GetDOFVelocityLimits()
            alim = robot.GetDOFAccelerationLimits()
            robot.SetDOFVelocityLimits(100 * vlim)
            robot.SetDOFAccelerationLimits(100 * alim)
            # Inverse dynamics
            qdd_full[active_dofs] = qdd
            robot.SetActiveDOFValues(q)
            robot.SetActiveDOFVelocities(qd)
            res = robot.ComputeInverseDynamics(qdd_full)
            # Restore vel/acc constraints
            robot.SetDOFVelocityLimits(vlim)
            robot.SetDOFAccelerationLimits(alim)
        return res[active_dofs]

    tau_max_ = robot.GetDOFTorqueLimits() * 4

    tau_max = np.vstack((-tau_max_[robot.GetActiveDOFIndices()], tau_max_[robot.GetActiveDOFIndices()])).T
    fs_coef = np.random.rand(dof) * 10
    pc_tau = constraint.JointTorqueConstraint(
        inv_dyn, tau_max, fs_coef, discretization_scheme=constraint.DiscretizationType.Interpolation)
    all_constraints = [pc_vel, pc_acc, pc_tau]
    # all_constraints = pc_vel

    instance = algo.TOPPRA(all_constraints, path, solver_wrapper='seidel')

    # Retime the trajectory, only this step is necessary.
    t0 = time.time()
    jnt_traj, _ = instance.compute_trajectory(0, 0)
    print("Parameterization time: {:} secs".format(time.time() - t0))
    ts_sample = np.linspace(0, jnt_traj.get_duration(), 100)
    qs_sample = jnt_traj.eval(ts_sample)
    qds_sample = jnt_traj.evald(ts_sample)
    qdds_sample = jnt_traj.evaldd(ts_sample)

    torque = []
    for q_, qd_, qdd_ in zip(qs_sample, qds_sample, qdds_sample):
        torque.append(inv_dyn(q_, qd_, qdd_) + fs_coef * np.sign(qd_))
    torque = np.array(torque)

    fig, axs = plt.subplots(dof, 1)
    for i in range(0, robot.GetActiveDOF()):
        axs[i].plot(ts_sample, torque[:, i])
        axs[i].plot([ts_sample[0], ts_sample[-1]], [tau_max[i], tau_max[i]], "--")
        axs[i].plot([ts_sample[0], ts_sample[-1]], [-tau_max[i], -tau_max[i]], "--")
    plt.xlabel("Time (s)")
    plt.ylabel("Torque $(Nm)$")
    plt.legend(loc='upper right')
    plt.show()

    # preview path
    for t in np.arange(0, jnt_traj.get_duration(), 0.01):
        robot.SetActiveDOFValues(jnt_traj.eval(t))
        time.sleep(0.01)  # 5x slow down

    # Compute the feasible sets and the controllable sets for viewing.
    # Note that these steps are not necessary.
    _, sd_vec, _ = instance.compute_parameterization(0, 0)
    X = instance.compute_feasible_sets()
    K = instance.compute_controllable_sets(0, 0)

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
