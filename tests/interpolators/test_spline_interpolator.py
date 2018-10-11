import numpy as np
import numpy.testing as npt
import pytest
from toppra import SplineInterpolator
try:
    import openravepy as orpy
    FOUND_OPENRAVE = True
except ImportError:
    FOUND_OPENRAVE = False


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
    yield robot
    env.Destroy()


class Test_SplineInterpolator(object):
    """ Test suite for :class:`SplineInterpolator`.
    """
    @pytest.mark.parametrize("sswp, wp, ss, path_interval", [
        [[0, 0.3, 0.5], [1,2,3], [0., 0.1, 0.2, 0.3, 0.5], [0, 0.5]],
        [np.r_[0, 0.3, 0.5], [1,2,3], [0.], [0, 0.5]]
    ])
    def test_scalar(self, sswp, wp, ss, path_interval):
        "A scalar (dof=1) trajectory."
        pi = SplineInterpolator(sswp, wp)  # 1 + 2s + 3s^2
        assert pi.dof == 1

        assert pi.eval(ss).shape == (len(ss), )
        assert pi.evald(ss).shape == (len(ss), )
        assert pi.evaldd(ss).shape == (len(ss), )
        assert pi.eval(0).shape == ()
        npt.assert_allclose(pi.get_path_interval(), path_interval)

    def test_5_dof(self):
        pi = SplineInterpolator([0, 1], np.random.rand(2, 5))
        # [1 + 2s + 3s^2]
        # [-2 + 3s + 4s^2 + 5s^3]

        ss = np.linspace(0, 1, 10)
        assert pi.dof == 5
        assert pi.eval(ss).shape == (10, 5)
        assert pi.evald(ss).shape == (10, 5)
        assert pi.evaldd(ss).shape == (10, 5)
        npt.assert_allclose(pi.get_path_interval(), np.r_[0, 1])

    def test_1waypoints(self):
        "The case where there is only one waypoint."
        pi = SplineInterpolator([0], [[1, 2, 3]])
        assert pi.dof == 3
        npt.assert_allclose(pi.get_path_interval(), np.r_[0, 0])
        npt.assert_allclose(pi.eval(0), np.r_[1, 2, 3])
        npt.assert_allclose(pi.evald(0), np.r_[0, 0, 0])

        npt.assert_allclose(pi.eval([0, 0]), [[1, 2, 3], [1, 2, 3]])
        npt.assert_allclose(pi.evald([0, 0]), [[0, 0, 0], [0, 0, 0]])

    @pytest.mark.parametrize("xs,ys, yd", [
        ([0, 1], [[0, 1], [2, 3]], [2, 2]),
        ([0, 2], [[0, 1], [0, 3]], [0, 1]),
    ])
    def test_2waypoints(self, xs, ys, yd):
        "There is only two waypoints. Linear interpolation is done between them."
        pi = SplineInterpolator(xs, ys, bc_type='natural')
        npt.assert_allclose(pi.get_path_interval(), xs)
        npt.assert_allclose(pi.evald((xs[0] + xs[1]) / 2), yd)
        npt.assert_allclose(pi.evaldd(0), np.zeros_like(ys[0]))

    @pytest.mark.skipif(not FOUND_OPENRAVE, reason="Not found openrave installation")
    @pytest.mark.parametrize("ss_waypoints, waypoints", [
        [[0, 0.2, 0.5, 0.9],  [[0.377, -0.369,  1.042, -0.265, -0.35 , -0.105, -0.74 ],
                              [ 1.131,  0.025,  0.778,  0.781,  0.543, -0.139,  0.222],
                              [-1.055,  1.721, -0.452, -0.268,  0.182, -0.935,  2.257],
                              [-0.274, -0.164,  1.492,  1.161,  1.958, -1.125,  0.567]]],
        [[0, 0.2], [[0.377, -0.369,  1.042, -0.265, -0.35 , -0.105, -0.74 ],
                    [ 1.131,  0.025,  0.778,  0.781,  0.543, -0.139,  0.222]]],
        [[0], [[0.377, -0.369,  1.042, -0.265, -0.35 , -0.105, -0.74 ]]]
    ])
    def test_compute_rave_trajectory(self, robot_fixture, ss_waypoints, waypoints):
        active_indices = robot_fixture.GetActiveDOFIndices()
        path = SplineInterpolator(ss_waypoints, waypoints)
        traj = path.compute_rave_trajectory(robot_fixture)
        spec = traj.GetConfigurationSpecification()

        xs = np.linspace(0, path.get_duration(), 10)

        # Interpolate with spline
        qs_spline = path.eval(xs)
        qds_spline = path.evald(xs)
        qdds_spline = path.evaldd(xs)

        # Interpolate with OpenRAVE
        qs_rave = []
        qds_rave = []
        qdds_rave = []
        for t in xs:
            data = traj.Sample(t)
            qs_rave.append(spec.ExtractJointValues(data, robot_fixture, active_indices, 0))
            qds_rave.append(spec.ExtractJointValues(data, robot_fixture, active_indices, 1))
            qdds_rave.append(spec.ExtractJointValues(data, robot_fixture, active_indices, 2))

        # Assert all close
        npt.assert_allclose(qs_spline, qs_rave)
        npt.assert_allclose(qds_spline, qds_rave)
        npt.assert_allclose(qdds_spline, qdds_rave)
