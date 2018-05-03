import openravepy as orpy
import toppra, time
import numpy as np
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A simple retiming example.")
    parser.add_argument('-e', '--env', help='OpenRAVE Environment file', default="data/lab1.env.xml")
    args = vars(parser.parse_args())
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
        print(traj_original.serialize())
        if traj_original is None:
            continue
        traj_retimed = toppra.retime_active_joints_kinematics(traj_original, robot)

        print("Original duration: {:.3f}. Retimed duration: {:3f}.".format(
            traj_original.GetDuration(), traj_retimed.GetDuration()))
        time.sleep(1)

        controller.SetPath(traj_retimed)
        robot.WaitForController(0)


