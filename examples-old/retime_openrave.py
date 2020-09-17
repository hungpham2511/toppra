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
        for i in range(traj_ra.dof):
            axs[0].plot(range(N), qs[:, i], style, c="C"+str(i), alpha=alpha)
            axs[1].plot(range(N), qds[:, i], style, c="C"+str(i), alpha=alpha)
            axs[2].plot(range(N), qdds[:, i], style, c="C"+str(i), alpha=alpha)

    plt.show()


def parse_cli_arguments():
    parser = argparse.ArgumentParser(description="A simple example in which trajectories, which are planned using"
                                                 "OpenRAVE is retimed using toppra. The trajectories are found using"
                                                 "birrt, the default planner. Goals are generated randomly and "
                                                 "uniformly.")
    parser.add_argument('-e', '--env', help='OpenRAVE Environment file', default="data/lab1.env.xml")
    parser.add_argument('-v', '--verbose', help='Show DEBUG log and plot trajectories', action="store_true")
    parser.add_argument('-N', '--Ngrid', help='Number of discretization step', default=100, type=int)
    args = vars(parser.parse_args())
    return args


def load_rave_robot(args):
    env = orpy.Environment()
    env.SetDebugLevel(0)
    env.Load(args['env'])
    env.SetViewer('qtosg')
    robot = env.GetRobots()[0]
    robot.SetDOFAccelerationLimits(np.ones(11) * 3)
    robot.SetActiveManipulator(robot.GetManipulators()[0])
    robot.SetActiveDOFs(robot.GetManipulators()[0].GetArmIndices())
    return robot


def generate_random_configuration(robot):
    lower, upper = robot.GetActiveDOFLimits()
    with robot:
        while True:
            qrand = np.random.rand(robot.GetActiveDOF()) * (upper - lower) + lower
            robot.SetActiveDOFValues(qrand)
            if robot.GetEnv().CheckCollision(robot) or robot.CheckSelfCollision():
                continue
            return qrand

def check_collision(robot, path):
    with robot.GetEnv():
        for q in np.linspace(0, path.duration, 100):
            robot.SetActiveDOFValues(path(q))
            if robot.GetEnv().CheckCollision(robot) or robot.CheckSelfCollision():
                print("Path in collision")

def main(env=None, test=False):
    "Main function."
    args = parse_cli_arguments()
    toppra.setup_logging('INFO')
    robot = load_rave_robot(args)
    basemanip = orpy.interfaces.BaseManipulation(robot)
    pc_torque = toppra.create_rave_torque_path_constraint(
        robot, discretization_scheme=toppra.constraint.DiscretizationType.Interpolation)

    it = 0
    while True or (test and it > 5):
        qrand = generate_random_configuration(robot)
        traj_original = basemanip.MoveActiveJoints(qrand, execute=False, outputtrajobj=True)
        if traj_original is None:
            continue

        traj_retimed, trajra = toppra.retime_active_joints_kinematics(
            traj_original, robot, output_interpolator=True, N=args['Ngrid'],
            additional_constraints=[pc_torque], solver_wrapper='seidel')

        print("Original duration: {:.3f}. Retimed duration: {:3f}.".format(
            traj_original.GetDuration(), traj_retimed.GetDuration()))

        robot.GetController().SetPath(traj_retimed)
        if args['verbose']:
            plot_trajectories(traj_original, traj_retimed, robot)

        robot.WaitForController(0)
        it += 1


if __name__ == '__main__':
    main()

