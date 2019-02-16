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
    vlim = np.vstack((-vlim_, vlim_)).T
    # Create acceleration bounds, then acceleration constraint object
    alim_ = robot.GetActiveDOFMaxAccel()
    alim = np.vstack((-alim_, alim_)).T
    pc_vel = constraint.JointVelocityConstraint(vlim)
    pc_acc = constraint.JointAccelerationConstraint(
        alim, discretization_scheme=constraint.DiscretizationType.Interpolation)

    # setup cartesian acceleration constraint to limit link 7
    # -0.5 <= a <= 0.5
    # cartersian acceleration
    def inverse_dynamics(q, qd, qdd):
        with robot:
            vlim_ = robot.GetDOFVelocityLimits()
            robot.SetDOFVelocityLimits(vlim_ * 1000)  # remove velocity limits to compute stuffs
            robot.SetActiveDOFValues(q)
            robot.SetActiveDOFVelocities(qd)

            qdd_full = np.zeros(robot.GetDOF())
            qdd_full[:qdd.shape[0]] = qdd

            accel_links = robot.GetLinkAccelerations(qdd_full)
            robot.SetDOFVelocityLimits(vlim_)
        return accel_links[6][:3]  # only return the translational components

    F_q = np.zeros((6, 3))
    F_q[:3, :3] = np.eye(3)
    F_q[3:, :3] = -np.eye(3)
    g_q = np.ones(6) * 0.5
    def F(q):
        return F_q
    def g(q):
        return g_q
    pc_cart_acc = constraint.CanonicalLinearSecondOrderConstraint(
        inverse_dynamics, F, g, dof=7)
    # cartesin accel finishes

    all_constraints = [pc_vel, pc_acc, pc_cart_acc]

    instance = algo.TOPPRA(all_constraints, path, solver_wrapper='seidel')

    # Retime the trajectory, only this step is necessary.
    t0 = time.time()
    jnt_traj, _ = instance.compute_trajectory(0, 0)
    print("Parameterization time: {:} secs".format(time.time() - t0))
    ts_sample = np.linspace(0, jnt_traj.get_duration(), 100)
    qs_sample = jnt_traj.eval(ts_sample)
    qds_sample = jnt_traj.evald(ts_sample)
    qdds_sample = jnt_traj.evaldd(ts_sample)

    cart_accel = []
    for q_, qd_, qdd_ in zip(qs_sample, qds_sample, qdds_sample):
        cart_accel.append(inverse_dynamics(q_, qd_, qdd_))
    cart_accel = np.array(cart_accel)

    plt.plot(ts_sample, cart_accel[:, 0], label="$a_x$")
    plt.plot(ts_sample, cart_accel[:, 1], label="$a_y$")
    plt.plot(ts_sample, cart_accel[:, 2], label="$a_z$")
    plt.plot([ts_sample[0], ts_sample[-1]], [0.5, 0.5], "--", c='black', label="Cart. Accel. Limits")
    plt.plot([ts_sample[0], ts_sample[-1]], [-0.5, -0.5], "--", c='black')
    plt.xlabel("Time (s)")
    plt.ylabel("Cartesian acceleration of the origin of link 6 $(m/s^2)$")
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
