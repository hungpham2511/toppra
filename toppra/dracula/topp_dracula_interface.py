from __future__ import print_function

import copy
import inspect
import math
import os

import toppra as ta
import toppra.constraint as constraint
import toppra.algorithm as algo
import numpy as np
import matplotlib.pyplot as plt
import argparse
import logging
logging.basicConfig()

# import torch
# from torch.utils import data

# from dataloader import ScoopDataset

_CALLS_TO_RUN_TOPP = 0

def RunTopp(knots_ext, vlim, alim,robot_command_rate, return_spline_parameters = False, topp_breaks_count = 1001):
    #robot_command_rate is not used if return_spline_parameters==True
    if not return_spline_parameters:
        assert(robot_command_rate)

    # print("To Pythonland!")
    _CALLS_TO_RUN_TOPP += 1 
    if _CALLS_TO_RUN_TOPP == 1: 
        # script_dirname = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
        script_getfile = inspect.getfile(inspect.currentframe()) # script filename (usually with path)
        # print("In TOPPRA Python:  dir(%s)  script(%s)" % (script_dirname, script_getfile))
        print("In TOPPRA Python call 1: script(%s)" % script_getfile)
    else:
        print("In TOPPRA Python call %d" % _CALLS_TO_RUN_TOPP)

    knots = copy.deepcopy(knots_ext)
    # print("Knots size: ", knots.shape)
    ta.setup_logging("INFO")  #causing issues?
    # Parameters
    N_samples = knots_ext.shape[0]
    dof = knots_ext.shape[1]

    # Random waypoints used to obtain a random geometric path. Here,
    # we use spline interpolation.
    # np.random.seed()
    # trajectory = np.random.randn(N_samples, dof)

    # training_set = ScoopDataset()
    # i_traj = np.random.randint(0, len(training_set.raw_json))
    # print("trajectory number: %d" % i_traj)
    # x_orig = np.array(training_set.raw_json[i_traj]['full_scoop'])
    #np.transpose(x_orig)
    # print("Knots start: ",knots[0,:])
    # print("Knots end: ",knots[-1,:])
    # print("Velocity bound: ", vlim)
    # print("Acceleration bound: ", alim)
    # plt.plot(trajectory)
    # plt.xlabel("idx")
    # plt.ylabel("Joint position (rad)")
    # plt.show()
    # N_samples=knots.shape[0]
    # dof = knots.shape[1]
    s_vector = np.linspace(0, 1, N_samples)
    #
    # print("num way points: %d" %N_samples)
    # print("num DOF: %d" %dof)
    if(N_samples<10):
        print("Error: Not enough knot points!")#exact number tbd
        return
    # print(s_vector.shape)
    # print(knots.shape[0])

    # print(knots)

    # print(np.sum(knots))
    # knots = knots + 1.0
    # knots = knots - 1.0
    # print(np.sum(knots))
    # print("knots +/-...")

    path = ta.SplineInterpolator(s_vector, knots)
    # print("yay, we made a path")
    # Create velocity bounds, then velocity constraint object
    # vlim_ = np.ones(dof) * 6.0 #np.random.rand(dof) * 20
    # vlim = np.vstack((-vlim_, vlim_)).T
    # Create acceleration bounds, then acceleration constraint object
    # alim_ = np.ones(dof) * 2.0 #np.random.rand(dof) * 2
    # alim_[6] = 2.0
    # alim = np.vstack((-alim_, alim_)).T
    # print(alim.shape)
    pc_vel = constraint.JointVelocityConstraint(vlim)
    pc_acc = constraint.JointAccelerationConstraint(
        alim, discretization_scheme=constraint.DiscretizationType.Interpolation)

    # Setup a parametrization instance with hot-qpOASES
    instance = algo.TOPPRA([pc_vel, pc_acc], path, gridpoints=np.linspace(0, 1, int(topp_breaks_count)),
                           solver_wrapper='hotqpoases')
    # print("yay we made an instance")
    X = instance.compute_feasible_sets()
    K = instance.compute_controllable_sets(0, 0)

    _, sd_vec, _ = instance.compute_parameterization(0, 0)

    X = np.sqrt(X)
    K = np.sqrt(K)

    # plt.plot(X[:, 0], c='green', label="Feasible sets")
    # plt.plot(X[:, 1], c='green')
    # plt.plot(K[:, 0], '--', c='red', label="Controllable sets")
    # plt.plot(K[:, 1], '--', c='red')
    # plt.plot(sd_vec, label="Velocity profile")
    # plt.title("Path-position path-velocity plot")
    # plt.xlabel("Path position")
    # plt.ylabel("Path velocity square")
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    # print("yay we are ready to compute the traj")

    jnt_traj, aux_traj = instance.compute_trajectory(0, 0, False, 'not-a-knot')
    # print(jnt_traj.cspl)
    '''
    compute_trajectory(self, sd_start, sd_end, return_profile=False, bc_type='not-a-knot')
    Parameters
    ----------
    sd_start: float
        Starting path velocity.
    sd_end: float
        Goal path velocity.

    jnt_traj is a <toppra.interpolator.SplineInterpolator> object, which is basically a wrapper of scipy.CubicSpline

    class SplineInterpolator(Interpolator):
    Attributes
    ----------
    dof : int
        Output dimension of the function
    cspl : :class:`scipy.interpolate.CubicSpline`
        The path.
    cspld : :class:`scipy.interpolate.CubicSpline`
        The path 1st derivative.
    cspldd : :class:`scipy.interpolate.CubicSpline`
        The path 2nd derivative.
    '''


    #modify
    if not return_spline_parameters:
        traj_duration = jnt_traj.get_duration()
        sample_number = int(math.ceil(traj_duration*robot_command_rate)+1)
        # print("just before printing traj duration... ")
        # print("trajectory duration:", traj_duration)
        # print("sample num:", sample_number)
        ts_sample = np.linspace(0, traj_duration, sample_number)
        qs_sample = jnt_traj.evaldd(ts_sample)
        # print(ts_sample)
        # print(qs_sample)
        # plt.plot(ts_sample, qs_sample)
        # plt.xlabel("Time (s)")
        # plt.ylabel("Joint acceleration (rad/s^2)")
        # plt.show()

        xs_sample = jnt_traj.eval(ts_sample)
        # xs_sample = copy.deepcopy(np.pad(xs_sample, ((10,10),(0,0)), 'edge'))

        # WHY pad it?
        # ??? ts_sample = copy.deepcopy(np.pad(ts_sample, (10, 10), 'edge')) # this is wrong... needs to be decreasing

        # print(xs_sample)
        # plot knot points with the first joint aligned:

        # plt.plot(ts_sample, xs_sample)
        # plt.xlabel("Time (s)")
        # plt.ylabel("Joint Position (rad)")
        # plt.show()


        # print(jnt_traj.eval(ts_sample))
        # x, y = training_set.__getitem__(i_traj)
        #
        # local_scoop = training_set.createScoopJson(xs_sample, y.cpu().numpy())
        # training_set.writeJson([local_scoop])
        #
        # print(qs_sample)
        # print("coeff shape", jnt_traj.cspl.c.shape)
        # print("break shape", jnt_traj.cspl.x.shape)
        # print("Done!")
        return (xs_sample, ts_sample, sample_number)

    else:
        #return knot points of the spline
        # print("cspl_x", jnt_traj.cspl.x.shape)
        # print("cspl_c", jnt_traj.cspl.c.shape)
        # print("Done!(spline)")
        return (jnt_traj.cspl.x, jnt_traj.cspl.c)


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
