import pytest
import toppra
import numpy as np
import openravepy as orpy


@pytest.fixture(scope='module')
def robot_fixture():
    env = orpy.Environment()
    env.Load("data/lab1.env.xml")
    robot = env.GetRobots()[0]
    manipulator = robot.GetManipulators()[0]
    robot.SetActiveDOFs(manipulator.GetArmIndices())
    # Generate IKFast if needed
    iktype = orpy.IkParameterization.Type.Transform6D
    ikmodel = orpy.databases.inversekinematics.InverseKinematicsModel(robot, iktype=iktype)
    if not ikmodel.load():
        print 'Generating IKFast {0}. It will take few minutes...'.format(iktype.name)
        ikmodel.autogenerate()
        print 'IKFast {0} has been successfully generated'.format(iktype.name)
    env.SetViewer('qtosg')
    yield robot
    env.Destroy()

@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_retime_kinematics(robot_fixture, seed):
    env = robot_fixture.GetEnv()
    basemanip = orpy.interfaces.BaseManipulation(robot_fixture)

    # Generate trajectory from seed
    arm_indices = robot_fixture.GetActiveDOFIndices()
    qlim_lower, qlim_upper = robot_fixture.GetActiveDOFLimits()
    qseed = np.random.randint(1000000, size=(1000,))  # Try 1000 times
    qsel = []
    for seed in qseed:
        np.random.seed(seed)
        qrand = qlim_lower + (qlim_upper - qlim_lower) * np.random.rand(len(arm_indices))
        with robot_fixture:
            robot_fixture.SetActiveDOFValues(qrand)
            incollision = env.CheckCollision(robot_fixture) or robot_fixture.CheckSelfCollision()
        if not incollision:
            qsel.append(qrand)
        if len(qsel) == 2:
            break
    assert len(qsel) == 2

    # Plan a path
    robot_fixture.SetActiveDOFValues(qsel[0])
    traj = basemanip.MoveActiveJoints(goal=qsel[1], execute=False, outputtrajobj=True)
    assert traj is not None

    # Retime it
    # Rave traj
    traj_new, interpolator = toppra.retime_active_joints_kinematics(traj, robot_fixture, output_interpolator=True, vmult=0.2)
    assert traj_new is not None
    robot_fixture.GetController().SetPath(traj_new)
    robot_fixture.WaitForController(0)

    # Start and initial waypoints

    # Consistent Joint Position
