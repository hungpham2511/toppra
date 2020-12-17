"""Unit test for the combined constraints interface test_run_toppra_jnt_crt."""

import os
import unittest

import numpy as np

from toppra import dracula


class TestRunToppJntCrt(unittest.TestCase):
    """Class for unit test."""
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    waypts_jnt = np.loadtxt("test_waypts_jnt_0.txt")
    lim_ones = np.ones(waypts_jnt.shape[1])
    lim_ones = np.vstack([-lim_ones, lim_ones]).T
    vlim_jnt = 1 * lim_ones  # rad/s
    alim_jnt = 7.5 * lim_ones  # rad/s^2
    waypts_crt = np.loadtxt("test_waypts_crt_0.txt")
    lim_ones = np.ones(waypts_crt.shape[1])
    lim_ones = np.vstack([-lim_ones, lim_ones]).T
    vlim_crt = 1 * lim_ones  # m/s
    alim_crt = 10 * lim_ones  # m/s^2

    def test_run_toppra_jnt_crt(self):
        """Test run_toppra_jnt_crt(), just assert no throw."""
        cs = dracula.run_toppra_jnt_crt(  # effectively assertNoRaise
            self.waypts_jnt,
            self.vlim_jnt,
            self.alim_jnt,
            self.waypts_crt,
            self.vlim_crt,
            self.alim_crt,
            return_cs=True,
        )
        v_abs = np.abs(cs.derivative(1)(cs.x))
        a_abs = np.abs(cs.derivative(2)(cs.x))
        self.assertAlmostEqual(v_abs[0].max(), 0)
        self.assertAlmostEqual(v_abs[-1].max(), 0)
        self.assertAlmostEqual(a_abs[0].max(), 0)
        self.assertAlmostEqual(a_abs[-1].max(), 0)
        self.assertGreater(cs.x[-1], 0.619)


if __name__ == "__main__":
    unittest.main()
