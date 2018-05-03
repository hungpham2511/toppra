import numpy as np
import numpy.testing as npt
from toppra import PolynomialPath


class Test_PolynomialInterpolator(object):
    """ Test suite for Polynomial Interpolator
    """

    def test_scalar(self):
        pi = PolynomialPath([1, 2, 3], s_start=0, s_end=2)  # 1 + 2s + 3s^2
        assert pi.dof == 1
        npt.assert_allclose(pi.eval([0, 0.5, 1]), [1, 2.75, 6])
        npt.assert_allclose(pi.evald([0, 0.5, 1]), [2, 5, 8])
        npt.assert_allclose(pi.evaldd([0, 0.5, 1]), [6, 6, 6])
        npt.assert_allclose(pi.get_path_interval(), np.r_[0, 2])

    def test_2_dof(self):
        pi = PolynomialPath([[1, 2, 3], [-2, 3, 4, 5]])
        # [1 + 2s + 3s^2]
        # [-2 + 3s + 4s^2 + 5s^3]
        assert pi.dof == 2
        npt.assert_allclose(
            pi.eval([0, 0.5, 1]), [[1, -2], [2.75, 1.125], [6, 10]])
        npt.assert_allclose(
            pi.evald([0, 0.5, 1]), [[2, 3], [5, 10.75], [8, 26]])
        npt.assert_allclose(pi.evaldd([0, 0.5, 1]), [[6, 8], [6, 23], [6, 38]])

