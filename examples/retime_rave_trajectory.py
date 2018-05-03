import openravepy as orpy
import toppra, time
import numpy as np
import argparse
import matplotlib.pyplot as plt


def plot_trajectories(traj1, traj2, robot, N=50):
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
        axs[1].plot([0, ts[-1]], [vel_max[i], vel_max[i]], style, c="C"+str(i))
        axs[1].plot([0, ts[-1]], [-vel_max[i], -vel_max[i]], style, c="C"+str(i))
        axs[2].plot([0, ts[-1]], [accel_max[i], accel_max[i]], style, c="C"+str(i))
        axs[2].plot([0, ts[-1]], [-accel_max[i], -accel_max[i]], style, c="C"+str(i))
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A simple retiming example.")
    parser.add_argument('-e', '--env', help='OpenRAVE Environment file', default="data/lab1.env.xml")
    parser.add_argument('-l', '--level', help='Logging level', default="INFO")
    parser.add_argument('-p', '--plot', help='Plot the original and the retimed trajectories', action='store_true')
    parser.add_argument('-n', '--Ngrid', help='Number of discretization step', default=500)
    args = vars(parser.parse_args())
    toppra.setup_logging(args['level'])
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

    while True:
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
            traj_original, robot, output_interpolator=True, N=args['Ngrid'])

        print("Original duration: {:.3f}. Retimed duration: {:3f}.".format(
            traj_original.GetDuration(), traj_retimed.GetDuration()))

        if args['plot']:
            plot_trajectories(traj_original, traj_retimed, robot)
        time.sleep(1)

        controller.SetPath(traj_retimed)
        robot.WaitForController(0)


