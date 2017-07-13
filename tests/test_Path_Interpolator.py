import numpy as np
import numpy.testing as npt

from toppra import PolynomialInterpolator, SplineInterpolator


class Test_PolynomialInterpolator(object):
    """ Test suite for Polynomial Interpolator
    """

    def test_scalar(self):
        pi = PolynomialInterpolator([1, 2, 3])  # 1 + 2s + 3s^2
        assert pi.dof == 1
        npt.assert_allclose(pi.eval([0, 0.5, 1]), [1, 2.75, 6])
        npt.assert_allclose(pi.evald([0, 0.5, 1]), [2, 5, 8])
        npt.assert_allclose(pi.evaldd([0, 0.5, 1]), [6, 6, 6])

    def test_2_dof(self):
        pi = PolynomialInterpolator([[1, 2, 3], [-2, 3, 4, 5]])
        # [1 + 2s + 3s^2]
        # [-2 + 3s + 4s^2 + 5s^3]
        assert pi.dof == 2
        npt.assert_allclose(
            pi.eval([0, 0.5, 1]), [[1, -2], [2.75, 1.125], [6, 10]])
        npt.assert_allclose(
            pi.evald([0, 0.5, 1]), [[2, 3], [5, 10.75], [8, 26]])
        npt.assert_allclose(pi.evaldd([0, 0.5, 1]), [[6, 8], [6, 23], [6, 38]])


class Test_SplineInterpolator(object):
    """ Test suite for Spline Interpolator
    """

    def test_scalar(self):
        pi = SplineInterpolator(np.linspace(0, 1, 3), [1, 2,
                                                       0])  # 1 + 2s + 3s^2

        ss = np.linspace(0, 1, 10)
        assert pi.dof == 1
        assert pi.eval(ss).shape == (10, )
        assert pi.evald(ss).shape == (10, )
        assert pi.evaldd(ss).shape == (10, )

    def test_5_dof(self):

        pi = SplineInterpolator([0, 1], np.random.rand(2, 5))
        # [1 + 2s + 3s^2]
        # [-2 + 3s + 4s^2 + 5s^3]

        ss = np.linspace(0, 1, 10)
        assert pi.dof == 5
        assert pi.eval(ss).shape == (10, 5)
        assert pi.evald(ss).shape == (10, 5)
        assert pi.evaldd(ss).shape == (10, 5)
