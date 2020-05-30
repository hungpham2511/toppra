import pytest
import numpy as np
import yaml
import re
import pandas
import tabulate
import time
try:
    import pathlib
except ImportError:
    import pathlib2 as pathlib

import toppra
import toppra.constraint as constraint
import toppra.algorithm as algo

import matplotlib.pyplot as plt


def test_robustness_main(request):
    """ Load problem suite based on regex, run test and report results.
    """
    toppra.setup_logging(request.config.getoption("--loglevel"))
    problem_regex = request.config.getoption("--robust_regex")
    visualize = request.config.getoption("--visualize")
    # parse problems from a configuration file
    parsed_problems = []
    path = pathlib.Path(__file__)
    path = path / '../problem_suite_1.yaml'
    problem_dict = yaml.load(path.resolve().read_text(), Loader=yaml.SafeLoader)
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
        t0 = time.time()
        path = toppra.SplineInterpolator(
            problem_data['ss_waypoints'],
            problem_data['waypoints'], bc_type='clamped')
        vlim = np.vstack((- problem_data['vlim'], problem_data['vlim'])).T
        alim = np.vstack((- problem_data['alim'], problem_data['alim'])).T
        pc_vel = constraint.JointVelocityConstraint(vlim)
        pc_acc = constraint.JointAccelerationConstraint(
            alim, discretization_scheme=constraint.DiscretizationType.Interpolation)
        t1 = time.time()

        if problem_data['desired_duration'] == 0:
            instance = algo.TOPPRA([pc_vel, pc_acc], path, gridpoints=problem_data['gridpoints'],
                                   solver_wrapper=problem_data['solver_wrapper'])
        else:
            instance = algo.TOPPRAsd([pc_vel, pc_acc], path, gridpoints=problem_data['gridpoints'],
                                     solver_wrapper=problem_data['solver_wrapper'])
            instance.set_desired_duration(problem_data['desired_duration'])

        t2 = time.time()
        jnt_traj = instance.compute_trajectory(0, 0)
        data = instance.problem_data
        t3 = time.time()

        if visualize:
            _t = np.linspace(0, jnt_traj.duration, 100)
            fig, axs = plt.subplots(2, 2)
            axs[0, 0].plot(data.K[:, 0], c="C0")
            axs[0, 0].plot(data.K[:, 1], c="C0")
            axs[0, 0].plot(data.sd_vec ** 2, c="C1")
            axs[0, 1].plot(_t, jnt_traj(_t))
            axs[1, 0].plot(_t, jnt_traj(_t, 1))
            axs[1, 1].plot(_t, jnt_traj(_t, 2))

            axs[0, 0].set_title("param")
            axs[0, 1].set_title("jnt. pos.")
            axs[1, 0].set_title("jnt. vel.")
            axs[1, 1].set_title("jnt. acc.")
            plt.show()

        if jnt_traj is None:
            all_success = False
            parsed_problems_df.loc[row_index, "status"] = "FAIL"
            parsed_problems_df.loc[row_index, "duration"] = None
        else:
            parsed_problems_df.loc[row_index, "status"] = "SUCCESS"
            parsed_problems_df.loc[row_index, "duration"] = jnt_traj.duration
        parsed_problems_df.loc[row_index, "t_init(ms)"] = (t1 - t0) * 1e3
        parsed_problems_df.loc[row_index, "t_setup(ms)"] = (t2 - t1) * 1e3
        parsed_problems_df.loc[row_index, "t_solve(ms)"] = (t3 - t2) * 1e3
    # get all rows with status different from NaN, then reports other columns.
    result_df = parsed_problems_df[parsed_problems_df["status"].notna()][
        ["status", "duration", "desired_duration", "name", "solver_wrapper",
         "nb_gridpoints", "problem_id", "t_init(ms)", "t_setup(ms)", "t_solve(ms)"]]
    result_df.to_csv('%s.result' % __file__)
    
    print("Test summary\n")
    print(tabulate.tabulate(result_df, result_df.columns))
    assert all_success, "Unable to solve some problems in the test suite"
