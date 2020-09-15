import copy
import inspect
import math
import logging

import toppra as ta
import toppra.constraint as constraint
import toppra.algorithm as algo
import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig()

from .zero_acceleration_start_end import ZeroAccelerationAtStartAndEnd


def RunTopp(knots_ext, vlim, alim,robot_command_rate, return_spline_parameters = False, topp_breaks_count = 1001, debug_active = False):
    #robot_command_rate is not used if return_spline_parameters==True
    if not return_spline_parameters:
        assert(robot_command_rate)

    knots = copy.deepcopy(knots_ext)
    if debug_active:
        print("Knots size: ", knots.shape)
    ta.setup_logging("INFO")  #causing issues?
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

    path = ta.SplineInterpolator(s_vector, knots, bc_type='clamped')
    if debug_active:
        print("yay, we made a path")
    pc_vel = constraint.JointVelocityConstraint(vlim)
    pc_acc = constraint.JointAccelerationConstraint(
        alim, discretization_scheme=constraint.DiscretizationType.Interpolation)

    instance = algo.TOPPRA([pc_vel, pc_acc], path, gridpoints=np.linspace(0, 1, int(topp_breaks_count)),solver_wrapper='seidel')
    if debug_active:
        print("yay we made an instance")
        X = instance.compute_feasible_sets()
        K = instance.compute_controllable_sets(0, 0)
        _, sd_vec, _ = instance.compute_parameterization(0, 0)
        X = np.sqrt(X)
        K = np.sqrt(K)
        plt.plot(X[:, 0], c='green', label="Feasible sets")
        plt.plot(X[:, 1], c='green')
        plt.plot(K[:, 0], '--', c='red', label="Controllable sets")
        plt.plot(K[:, 1], '--', c='red')
        plt.plot(sd_vec, label="Velocity profile")
        plt.title("Path-position path-velocity plot")
        plt.xlabel("Path position")
        plt.ylabel("Path velocity square")
        plt.legend()
        plt.tight_layout()
        plt.show()

    if debug_active:
        print("yay we are ready to compute the traj")
    jnt_traj = instance.compute_trajectory(0, 0)
    if debug_active:
        print(jnt_traj.cspl)
    if debug_active:
    	csplcp = copy.deepcopy(jnt_traj.cspl)

    ZeroAccelerationAtStartAndEnd(jnt_traj.cspl)

    #modify
    if not return_spline_parameters:
        # return samples
        traj_duration = jnt_traj.cspl.x[-1]
        sample_number = int(math.ceil(traj_duration*robot_command_rate)+1)
        print("just before printing traj duration... ")
        print("trajectory duration:", traj_duration)
        print("sample num:", sample_number)
        ts_sample = np.linspace(0, traj_duration, sample_number)
        
        xs_sample = jnt_traj.cspl(ts_sample)
        xds_sample = jnt_traj.cspl(ts_sample,1)
        xdds_sample = jnt_traj.cspl(ts_sample,2)
        if debug_active:
            print(ts_sample)
            print(qs_sample)
            plt.plot(ts_sample, qs_sample)
            plt.xlabel("Time (s)")
            plt.ylabel("Joint acceleration (rad/s^2)")
            plt.show()

        if debug_active:
            return (xs_sample, ts_sample, sample_number, xds_sample, xdds_sample)
        else:
            return (np.ascontiguousarray(xs_sample, dtype=np.float64),
                    np.ascontiguousarray(ts_sample, dtype=np.float64),
                    sample_number)

    else:
        # return knot points of the spline
        if debug_active:
            print("cspl_x", jnt_traj.cspl.x.shape)
            print("cspl_c", jnt_traj.cspl.c.shape)
            print("Done!(spline)")

        topp_breaks_count_final = int(len(jnt_traj.cspl.x))
        if debug_active:
            return (jnt_traj.cspl.x, jnt_traj.cspl.c, jnt_traj.cspl, csplcp)
        else:
            return (topp_breaks_count_final, 
                    np.ascontiguousarray(jnt_traj.cspl.x, dtype=np.float64), 
                    np.ascontiguousarray(jnt_traj.cspl.c, dtype=np.float64))

if __name__ == '__main__':
    for i in range(0,1):
        m = np.random.rand(55,7)
        vlim = np.asarray([2]*7)
        vlim = np.transpose(np.vstack((-vlim, vlim)))
        alim = np.asarray([2]*7)
        alim = np.transpose(np.vstack((-alim, alim)))
        robot_command_rate = 20
        RunTopp(m, vlim, alim, robot_command_rate,False)
        RunTopp(m, vlim, alim, robot_command_rate,True)
