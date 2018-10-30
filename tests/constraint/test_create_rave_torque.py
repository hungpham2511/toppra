import pytest
import toppra
import numpy as np


@pytest.fixture(scope="module")
def barret_robot(rave_env):
    rave_env.Reset()
    rave_env.Load("data/lab1.env.xml")
    robot = rave_env.GetRobots()[0]
    yield robot


@pytest.mark.parametrize("dof", [3, 5, 7])
def test_torque_bound_barret(barret_robot, dof):
    barret_robot.SetActiveDOFs(range(dof))
    constraint = toppra.create_rave_torque_path_constraint(barret_robot)
    np.random.seed(0)
    path = toppra.SplineInterpolator(np.linspace(0, 1, 5), np.random.rand(5, dof))
    a, b, c, F, g, _, _ = constraint.compute_constraint_params(
        path, np.linspace(0, path.get_duration(), 5), 1.0)

    assert a.shape[1] == dof
    assert b.shape[1] == dof
    assert c.shape[1] == dof
    assert F.shape[1:] == (2 * dof, dof)
    assert g.shape[1] == 2 * dof


    
