import pdb
import copy
import inspect
import math
import logging

# import toppra as ta
# import toppra.constraint as constraint
# import toppra.algorithm as algo

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splprep, BSpline

logging.basicConfig()

# from .zero_acceleration_start_end import ZeroAccelerationAtStartAndEnd

def RunTopp(knots_ext, topp_breaks_count = None, debug_active = False):
    #robot_command_rate is not used if return_spline_parameters==True
    global _CALLS_TO_RUN_TOPP 

    knots = copy.deepcopy(knots_ext)
    if debug_active:
        print("Knots size: ", knots.shape)

    # Parameters
    N_samples = knots_ext.shape[0]
    dof = knots_ext.shape[1]

    s_vector = np.linspace(0, 1, N_samples)
    if debug_active:
        print("num way points: %d" %N_samples)
        print("num DOF: %d" %dof)
    if(N_samples<10):
        print("Error: Not enough knot points!")#exact number tbd
        return

    if topp_breaks_count is None:
        topp_breaks_count = N_samples
        ret_all_t = s_vector
    else:
        topp_breaks_count = int(topp_breaks_count)
        ret_all_t = np.linspace(0, 1, topp_breaks_count, dtype=np.float64)

    ret_all_x = np.zeros((topp_breaks_count, dof), dtype=np.float64)
    
    for t_dof in range(dof):
        t_bspline_tck = splrep(x=s_vector, y=knots[:, t_dof].T, k=3)
        t_bspline = BSpline(*t_bspline_tck)
        ret_all_x[:, t_dof] = t_bspline(ret_all_t)

    if debug_active:
        print("yay, we fit B-splines")
        for t_dof in range(dof):
            plt.plot(ret_all_t, ret_all_x[:, t_dof], c='green', label="Feasible sets")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return (topp_breaks_count, 
                    np.ascontiguousarray(ret_all_x, dtype=np.float64))

if __name__ == '__main__':
    for i in range(0,1):
        m = np.random.rand(55,7)
        vlim = np.asarray([2]*7)
        vlim = np.transpose(np.vstack((-vlim, vlim)))
        alim = np.asarray([2]*7)
        alim = np.transpose(np.vstack((-alim, alim)))
        robot_command_rate = 20
        RunTopp(m, debug_active=False)
        RunTopp(m, 50, debug_active=True)
