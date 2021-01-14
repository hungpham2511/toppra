"""Unit test for the combined constraints interface test_run_topp_jnt_crt."""

import glob
import os
import unittest

import numpy as np

from toppra import dracula


class TestRunToppJntCrt(unittest.TestCase):
    """Class for unit test."""

    paths = sorted(
        glob.glob(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "test_waypts_crt_*.txt",
            )
        )
    )
    waypts_jnt_list = [
        np.loadtxt(path.replace("_crt_", "_jnt_")) for path in paths
    ]
    waypts_crt_list = [np.loadtxt(path) for path in paths]
    lim_ones = np.ones(waypts_jnt_list[0].shape[1])
    lim_ones = np.vstack([-lim_ones, lim_ones]).T
    vlim_jnt = 1 * lim_ones  # rad/s
    alim_jnt = 7.5 * lim_ones  # rad/s^2
    lim_ones = np.ones(waypts_crt_list[0].shape[1])
    lim_ones = np.vstack([-lim_ones, lim_ones]).T
    vlim_crt = 1 * lim_ones  # m/s
    alim_crt = 10 * lim_ones  # m/s^2

    def test_run_topp_jnt_crt(self):
        """Test run_topp_jnt_crt(), just assert no throw."""
        for i, (waypts_jnt, waypts_crt) in enumerate(
            zip(self.waypts_jnt_list, self.waypts_crt_list)
        ):
            print(f"Testing data file {i}...")
            cs = dracula.run_topp_jnt_crt(  # effectively assertNoRaise
                waypts_jnt,
                self.vlim_jnt,
                self.alim_jnt,
                waypts_crt,
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
            self.assertGreater(cs.x[-1], 0.479)


if __name__ == "__main__":
    unittest.main()
