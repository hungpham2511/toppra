import pytest
import toppra
import numpy as np
try:
    import openravepy as orpy
    FOUND_OPENRAVE = True
except ImportError:
    FOUND_OPENRAVE = False

toppra.setup_logging("DEBUG")

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
        print('Generating IKFast {0}. It will take few minutes...'.format(iktype.name))
        ikmodel.autogenerate()
        print('IKFast {0} has been successfully generated'.format(iktype.name))
    # env.SetViewer('qtosg')
    toppra.setup_logging("INFO")
    yield robot
    env.Destroy()


@pytest.mark.skipif(not FOUND_OPENRAVE, reason="Not found openrave installation")
@pytest.mark.parametrize("seed", range(90, 100), ids=["Seed=" + str(i) for i in range(90, 100)])
@pytest.mark.parametrize("solver_wrapper", ["hotqpoases", "seidel"])
def test_retime_kinematics_ravetraj(robot_fixture, seed, solver_wrapper):
    env = robot_fixture.GetEnv()
    basemanip = orpy.interfaces.BaseManipulation(robot_fixture)

    # Generate random trajectory from seed
    arm_indices = robot_fixture.GetActiveDOFIndices()
    qlim_lower, qlim_upper = robot_fixture.GetActiveDOFLimits()
    np.random.seed(seed)
    qsel = []
    for _ in range(1000):
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

    # Rave traj
    traj_new, interpolator = toppra.retime_active_joints_kinematics(
        traj, robot_fixture, output_interpolator=True, vmult=1.0, solver_wrapper=solver_wrapper)
    assert traj_new is not None
    assert traj_new.GetDuration() < traj.GetDuration() + 1  # Should not be too far from the optimal value


@pytest.mark.skipif(not FOUND_OPENRAVE, reason="Not found openrave installation")
@pytest.mark.parametrize("seed", range(100, 110), ids=["Seed="+str(i) for i in range(100, 110)])
@pytest.mark.parametrize("solver_wrapper", ["hotqpoases", "seidel", "ecos"])
def test_retime_kinematics_waypoints(robot_fixture, seed, solver_wrapper):
    dof = robot_fixture.GetActiveDOF()

    # Generate random trajectory from seed
    qlim_lower, qlim_upper = robot_fixture.GetActiveDOFLimits()

    np.random.seed(seed)
    waypoints = np.random.rand(5, dof)
    for i in range(5):
        waypoints[i] = qlim_lower + waypoints[i] * (qlim_upper - qlim_lower)

    traj_new, trajra = toppra.retime_active_joints_kinematics(
        waypoints, robot_fixture, output_interpolator=True, solver_wrapper=solver_wrapper)
    assert traj_new is not None
    assert traj_new.GetDuration() < 30 and traj_new.GetDuration() > 0

