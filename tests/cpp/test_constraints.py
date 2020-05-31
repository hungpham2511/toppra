import pytest
import toppra.cpp as tac

pytestmark = pytest.mark.skipif(
    not tac.bindings_loaded(), reason="c++ bindings not built"
)

@pytest.fixture
def path():
    from numpy import array
    qs = [  array([ 2.13637886, -0.66327522,  1.77786313,  1.87420341,  2.58514541, -1.8993774 ]),
            array([ 1.99090939, -0.55557809,  1.63229007,  1.80377687,  2.4600473 , -1.77512358]),
            array([ 1.81337749, -0.5155883 ,  1.70778401,  1.85440536,  2.35954536, -1.57885839]),
            ]
    vs = [  array([ 0.,  0.,  0.,  0.,  0.,  0.]),
            array([-1.07667124,  0.49228973, -0.23359707, -0.0659935 , -0.75200016, 1.06839671]),
            array([ 0.,  0.,  0.,  0.,  0.,  0.])
            ]
    ts = [0, 0.3, 0.6]
    from toppra.cpp import PiecewisePolyPath
    yield PiecewisePolyPath.constructHermite(qs, vs, ts)

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

def test_joint_torque_pinocchio(path):
    try:
        import pinocchio
    except ImportError as err:
        print(err)
        return

    model = pinocchio.buildSampleModelManipulator()
    jt = tac.jointTorque.Pinocchio(model)

    a0, b0, c0, F0, g0, u0, x0 = jt.computeParams(path, [ path.path_interval[0], ])
    assert u0 is None
    assert x0 is None

    # Check that Python will keep value the model.
    del model

    a1, b1, c1, F1, g1, u1, x1 = jt.computeParams(path, [ path.path_interval[0], ])
    assert u1 is None
    assert x1 is None

    assert all([ (e0 == e1).all() for e0,e1 in zip(a0,a1) ])
    assert all([ (e0 == e1).all() for e0,e1 in zip(b0,b1) ])
    assert all([ (e0 == e1).all() for e0,e1 in zip(c0,c1) ])
    assert all([ (e0 == e1).all() for e0,e1 in zip(F0,F1) ])
    assert all([ (e0 == e1).all() for e0,e1 in zip(g0,g1) ])
