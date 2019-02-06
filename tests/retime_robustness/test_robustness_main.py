import pytest
import numpy as np
import yaml
import re
import pandas

import toppra
import toppra.constraint as constraint
import toppra.algorithm as algo

# toppra.setup_logging(level="DEBUG")

# NOTE: Select problems to test with this regex.
problem_regex = ".*oa.*"


def test_robustness_main():
    """ Load problem suite based on regex, run test and report results.
    """
    # parse problems from a configuration file
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
                    parsed_problems.append({
                        "name": key,
                        "problem_id": "{:}-{:5f}-{:}-{:}".format(key, duration, solver_wrapper, nb_gridpoints),
                        'waypoints': np.array(problem_dict[key]['waypoints'], dtype=float),
                        'ss_waypoints': ss_waypoints,
                        'vlim': np.r_[problem_dict[key]['vlim']],
                        'alim': np.r_[problem_dict[key]['alim']],
                        'desired_duration': duration,
                        'solver_wrapper': solver_wrapper,
                        'gridpoints': np.linspace(ss_waypoints[0], ss_waypoints[-1], nb_gridpoints),
                        'nb_gridpoints': nb_gridpoints
                    })
    parsed_problems_df = pandas.DataFrame(parsed_problems)

    # solve problems that matched the given regex
    all_success = True
    for row_index, problem_data in parsed_problems_df.iterrows():
        if re.match(problem_regex, problem_data['problem_id']) is None:
            continue
        path = toppra.SplineInterpolator(
            problem_data['ss_waypoints'],
            problem_data['waypoints'])
        vlim = np.vstack((- problem_data['vlim'], problem_data['vlim'])).T
        alim = np.vstack((- problem_data['alim'], problem_data['alim'])).T
        pc_vel = constraint.JointVelocityConstraint(vlim)
        pc_acc = constraint.JointAccelerationConstraint(
            alim, discretization_scheme=constraint.DiscretizationType.Interpolation)

        if problem_data['desired_duration'] == 0:
            instance = algo.TOPPRA([pc_vel, pc_acc], path, gridpoints=problem_data['gridpoints'],
                                   solver_wrapper=problem_data['solver_wrapper'])
        else:
            instance = algo.TOPPRAsd([pc_vel, pc_acc], path, gridpoints=problem_data['gridpoints'],
                                     solver_wrapper=problem_data['solver_wrapper'])
            instance.set_desired_duration(problem_data['desired_duration'])

        jnt_traj, aux_traj, data = instance.compute_trajectory(0, 0, return_data=True)

        if jnt_traj is None:
            all_success = False
            parsed_problems_df.loc[row_index, "status"] = "FAIL"
            parsed_problems_df.loc[row_index, "duration"] = None
        else:
            parsed_problems_df.loc[row_index, "status"] = "SUCCESS"
            parsed_problems_df.loc[row_index,
                                   "duration"] = jnt_traj.get_duration()

    # get all rows with status different from NaN, then reports other columns.
    result_df = parsed_problems_df[parsed_problems_df["status"].notna()][
        ["status", "duration", "desired_duration", "name", "solver_wrapper", "nb_gridpoints", "problem_id"]]
    print("\n")
    print(result_df)
    assert all_success, "Unable to solve some problems in the test suite"
