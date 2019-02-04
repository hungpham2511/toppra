import pytest
import numpy as np
import numpy.testing as npt
import yaml
import re

import toppra
import toppra.constraint as constraint
import toppra.algorithm as algo


toppra.setup_logging(level="INFO")

problem_regex = "prob1.*oa.*"


def test_robustness_main():
    """ Load problem suite and test all.
    """

    # parse all problems from configuration file
    parsed_problems = []
    with open("retime_robustness/problem_suite_1.yaml", "r") as f:
        problem_dict = yaml.load(f.read())
    for key in problem_dict:
        if len(problem_dict[key]['ss_waypoints']) == 2:
            ss_waypoints = np.linspace(problem_dict[key]['ss_waypoints'][0],
                                       problem_dict[key]['ss_waypoints'][1],
                                       len(problem_dict[key]['waypoints']))

        for duration in problem_dict[key]['desired_duration']:
            for solver_wrapper in problem_dict[key]['solver_wrapper']:
                for nb_gridpoints in problem_dict[key]['nb_gridpoints']:
                    parsed_problems.append([
                        "{:}-{:5f}-{:}-{:}".format(key, duration, solver_wrapper, nb_gridpoints),
                        np.array(problem_dict[key]['waypoints'], dtype=float),
                        ss_waypoints,
                        np.r_[problem_dict[key]['vlim']],
                        np.r_[problem_dict[key]['alim']],
                        duration,
                        solver_wrapper,
                        np.linspace(ss_waypoints[0], ss_waypoints[-1], nb_gridpoints)
                    ])

    # attempt to solve
    all_success = True
    all_res = []
    for problem_data in parsed_problems:
        if re.match(problem_regex, problem_data[0]) is None:
            continue
        path = toppra.SplineInterpolator(problem_data[2], problem_data[1])
        vlim = np.vstack((- problem_data[3], problem_data[3])).T
        alim = np.vstack((- problem_data[3], problem_data[3])).T
        pc_vel = constraint.JointVelocityConstraint(vlim)
        pc_acc = constraint.JointAccelerationConstraint(
            alim, discretization_scheme=constraint.DiscretizationType.Interpolation)

        instance = algo.TOPPRA([pc_vel, pc_acc], path, gridpoints=problem_data[7],
                               solver_wrapper=problem_data[6])

        jnt_traj, aux_traj, data = instance.compute_trajectory(0, 0, return_data=True)

        if jnt_traj is None:
            all_success = False
            res_string = "FAIL             {:}".format(problem_data[0])
        else:
            res_string = "SUCCESS {:5f} {:}".format(jnt_traj.get_duration(), problem_data[0])
        all_res.append(res_string)

    print("\nTest results:")
    for res in all_res:
        print(res)
    assert all_success, "Unable to solve some problems in the test suite"
        

