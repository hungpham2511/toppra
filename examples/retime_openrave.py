import openravepy as orpy
import toppra, time
import numpy as np
import argparse
import matplotlib.pyplot as plt


def plot_trajectories(traj1, traj2, robot, N=50):
    "Plot the trajectory (position, velocity and acceleration)."
    fig, axs = plt.subplots(3, 1, sharex=True)
    for idx, traj in enumerate([traj1, traj2]):
        ts = np.linspace(0, traj.GetDuration(), N)
        traj_ra = toppra.RaveTrajectoryWrapper(traj, robot)
        qs = traj_ra.eval(ts)
        qds = traj_ra.evald(ts)
        qdds = traj_ra.evaldd(ts)

        if idx == 0:
            style = "o"
            alpha = 0.5
        else:
            style = "-"
            alpha = 1.0
        for i in range(traj_ra.get_dof()):
            axs[0].plot(ts, qs[:, i], style, c="C"+str(i), alpha=alpha)
            axs[1].plot(ts, qds[:, i], style, c="C"+str(i), alpha=alpha)
            axs[2].plot(ts, qdds[:, i], style, c="C"+str(i), alpha=alpha)

    vel_max = robot.GetActiveDOFMaxVel()
    accel_max = robot.GetActiveDOFMaxAccel()
    style = "--"
    for i in range(traj_ra.get_dof()):
        axs[1].plot([0, ts[-1]], [vel_max[i], vel_max[i]], style, c="C" + str(i))
        axs[1].plot([0, ts[-1]], [-vel_max[i], -vel_max[i]], style, c="C" + str(i))
        axs[2].plot([0, ts[-1]], [accel_max[i], accel_max[i]], style, c="C" + str(i))
        axs[2].plot([0, ts[-1]], [-accel_max[i], -accel_max[i]], style, c="C" + str(i))
    plt.show()


def main(env=None, test=False):
    "Main function."
    parser = argparse.ArgumentParser(description="A simple example in which trajectories, which are planned using"
                                                 "OpenRAVE is retimed using toppra. The trajectories are found using"
                                                 "birrt, the default planner. Goals are generated randomly and "
                                                 "uniformly.")
    parser.add_argument('-e', '--env', help='OpenRAVE Environment file', default="data/lab1.env.xml")
    parser.add_argument('-v', '--verbose', help='Show DEBUG log and plot trajectories', action="store_true")
    parser.add_argument('-N', '--Ngrid', help='Number of discretization step', default=100, type=int)
    args = vars(parser.parse_args())

    if args['verbose']:
        toppra.setup_logging('DEBUG')
    else:
        toppra.setup_logging('INFO')

    if env is None:
        env = orpy.Environment()
    env.SetDebugLevel(0)
    env.Load(args['env'])
    env.SetViewer('qtosg')
    robot = env.GetRobots()[0]
    robot.SetDOFAccelerationLimits(np.ones(11) * 3)
    manipulator = robot.GetManipulators()[0]
    robot.SetActiveManipulator(manipulator)
    robot.SetActiveDOFs(manipulator.GetArmIndices())
    controller = robot.GetController()
    basemanip = orpy.interfaces.BaseManipulation(robot)
    dof = robot.GetActiveDOF()

    pc_torque = toppra.create_rave_torque_path_constraint(
        robot, discretization_scheme=toppra.constraint.DiscretizationType.Interpolation)

    it = 0
    while True or (test and it > 5):
        lower, upper = robot.GetActiveDOFLimits()
        qrand = np.random.rand(dof) * (upper - lower) + lower
        with robot:
            robot.SetActiveDOFValues(qrand)
            incollision = env.CheckCollision(robot) or robot.CheckSelfCollision()
            if incollision:
                continue
        traj_original = basemanip.MoveActiveJoints(qrand, execute=False, outputtrajobj=True)
        if traj_original is None:
            continue
        traj_retimed, trajra = toppra.retime_active_joints_kinematics(
            traj_original, robot, output_interpolator=True, N=args['Ngrid'],
            additional_constraints=[pc_torque], solver_wrapper='seidel')

        print("Original duration: {:.3f}. Retimed duration: {:3f}.".format(
            traj_original.GetDuration(), traj_retimed.GetDuration()))

        if args['verbose']:
            plot_trajectories(traj_original, traj_retimed, robot)
        time.sleep(1)

        controller.SetPath(traj_retimed)
        robot.WaitForController(0)
        it += 1


if __name__ == '__main__':
    main()

