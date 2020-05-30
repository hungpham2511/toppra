import pytest
import toppra.cpp as tac

pytestmark = pytest.mark.skipif(
    not tac.bindings_loaded(), reason="c++ bindings not built"
)

def test_linear_vel():
    c = tac.LinearJointVelocity([-1, -1], [1, 1])
    c.discretizationType = tac.DiscretizationType.Interpolation

    assert not c.hasUbounds
    assert c.hasXbounds
    assert not c.hasLinearInequalities


def test_linear_accel():
    c = tac.LinearJointAcceleration([-1, -1], [1, 1])
    c.discretizationType = tac.DiscretizationType.Interpolation

    assert not c.hasUbounds
    assert not c.hasXbounds
    assert c.hasLinearInequalities
