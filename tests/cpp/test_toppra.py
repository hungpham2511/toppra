import pytest
import numpy as np
import toppra.cpp as tac

pytestmark = pytest.mark.skipif(
    not tac.bindings_loaded(), reason="c++ bindings not built"
)

@pytest.fixture
def path():
    c = np.array([
        [-0.500000, -0.500000, 1.500000, 0.500000, 0.000000, 3.000000, 0.000000, 0.000000],
        [-0.500000, -0.500000, 0.000000, -1.000000, 1.500000, 2.500000, 1.000000, 3.000000],
        [-0.500000, -0.500000, -1.500000, -2.500000, 0.000000, -1.000000, 2.000000, 4.000000]
    ])
    c = c.reshape((3, 4, 2))
    p = tac.PiecewisePolyPath(c, [0, 1, 2, 3])
    yield p


def test_solve_toppra(path):
    cv = tac.LinearJointVelocity([-1, -1], [1, 1])
    ca = tac.LinearJointAcceleration([-0.2, -0.2], [0.2, 0.2])
    prob = tac.TOPPRA([cv, ca], path)
    prob.setN(50)
    ret = prob.computePathParametrization()
    assert ret == tac.ReturnCode.OK
    data = prob.parametrizationData
    sd_expected = [0., 0.00799999, 0.01559927, 0.02295854, 0.03021812,
                   0.0375065, 0.04494723, 0.05266502, 0.06079176, 0.06947278, 0.07887417, 0.08890758,
                   0.08734253, 0.08331795, 0.07962036, 0.07621324, 0.0730652, 0.07014912, 0.06744149,
                   0.06492187, 0.06257243, 0.06037763, 0.05832396, 0.05639983, 0.05459562,
                   0.05290406, 0.05132157, 0.04985237, 0.04852316, 0.04745693, 0.04761904, 0.0285715,
                   0.05376003, 0.04275653, 0.04126188, 0.04013804, 0.03912958, 0.03818766,
                   0.03729606, 0.0364472, 0.03563649, 0.03486069, 0.03411724, 0.03340395, 0.03271895,
                   0.03206054, 0.02268897, 0.01495547, 0.00883489, 0.00394282, 0.]
    np.testing.assert_allclose(data.parametrization, sd_expected, atol=1e-6)
