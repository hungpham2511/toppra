from toppra import (create_velocity_path_constraint,
                    create_acceleration_path_constraint,
                    qpOASESPPSolver,
                    compute_trajectory_gridpoints,
                    interpolate_constraint)
from rave.Rave import SplineInterpolator
import numpy as np
import matplotlib.pyplot as plt
import time
from multi_contacts import smooth_singularities
np.set_printoptions(3)


# REP: number of repetition
REP = 1
SAVE_FIG = False
SHOW = True
N = 100
N_knot = 5
SEED = 9
dof = 7


def generate_n_dof_test_case(n, N):
    """Generate a random n-dof path with velocity and acceleration
    bounds.

    Args:
    -----
        n: An Int. dof of the path.

    Returns:
    --------
        constraints: A List of PathConstraint.
        data: A tuple. Contains necessary data to recreate this instance.
    """
    np.random.seed(SEED)
    way_pts = np.random.randn(N_knot, n)
    pi = SplineInterpolator(np.linspace(0, 1, 5), way_pts)
    ss = np.linspace(0, 1, N + 1)
    # Velocity Constraint
    vlim_ = np.random.rand(n) * 20
    vlim = np.vstack((-vlim_, vlim_)).T
    pc_vel = create_velocity_path_constraint(pi, ss, vlim)
    # Acceleration Constraints
    alim_ = np.random.rand(n) * 2
    alim = np.vstack((-alim_, alim_)).T
    pc_acc = create_acceleration_path_constraint(pi, ss, alim)
    constraints = [pc_vel, pc_acc]
    data = (way_pts, alim_, vlim_)
    print """
Generation:
-----------
   - generated {0} random points
   - velocity bounds: {1}
   - acceleration bounds: {2}
   - Number of discretized points (RA): {3}
""".format(N_knot, vlim_, alim_, N)
    return constraints, pi, data


if __name__ == "__main__":
    cset, path, data = generate_n_dof_test_case(dof, N)
    print """
Report:
------ """
    # collocation strategy
    t_ = time.time()
    pp = qpOASESPPSolver(cset)
    t_setup = time.time() - t_
    us, xs = pp.solve_topp(save_solutions=False)
    t_solve = time.time() - t_ - t_setup
    t, q, qd, qdd = compute_trajectory_gridpoints(path, pp.ss, us, xs)
    t_total = time.time() - t_

    # colloc with smoothing
    us_smth, xs_smth = smooth_singularities(pp, us, xs)
    t_smth, q_smth, qd_smth, qdd_smth = compute_trajectory_gridpoints(
        path, pp.ss, us_smth, xs_smth)

    # interpolation strategy
    t_ = time.time()
    cset_intp = [interpolate_constraint(pc) for pc in cset]
    t_intp_interpolate = time.time() - t_
    pp_intp = qpOASESPPSolver(cset_intp)
    t_intp_setup = time.time() - t_
    us_intp, xs_intp = pp_intp.solve_topp()
    t_intp_solve = time.time() - t_ - t_intp_setup
    t_intp, q_intp, qd_intp, qdd_intp = compute_trajectory_gridpoints(
        path, pp_intp.ss, us_intp, xs_intp)
    t_intp_total = time.time() - t_

    print """
Total computation time
(colloc.)      : {:8.4f} msec
    -     setup: {:8.4f} msec
    -     solve: {:8.4f} msec

(intp.)        : {:8.4f} msec
    -     setup: {:8.4f} msec
    -     interpolate/setup: {:8.4f} msec
    -     solve: {:8.4f} msec

""".format(1000 * t_total, 1000 * t_setup, 1000 * t_solve, 1000 *
           t_intp_total, 1000 * t_intp_setup, 1e3 *
           t_intp_interpolate, 1000 * t_intp_solve)

    if SHOW:
        # plotting
        f, axs = plt.subplots(3, 3, figsize=[9, 6])
        axs[0, 0].plot(t, qd[:, [1, 2]])
        axs[0, 0].plot(t, qd, alpha=0.2)
        axs[1, 0].plot(t, qdd[:, [1, 2]])
        axs[1, 0].plot(t, qdd, alpha=0.2)
        axs[2, 0].plot(np.sqrt(pp.K[:, 0]), '--', c='C3')
        axs[2, 0].plot(np.sqrt(pp.K[:, 1]), '--', c='C3')
        axs[2, 0].plot(np.sqrt(xs))
        axs[0, 1].plot(t_smth, qd_smth[:, [1, 2]])
        axs[0, 1].plot(t_smth, qd_smth, alpha=0.2)
        axs[1, 1].plot(t_smth, qdd_smth[:, [1, 2]])
        axs[1, 1].plot(t_smth, qdd_smth, alpha=0.2)
        axs[2, 1].plot(np.sqrt(pp.K[:, 0]), '--', c='C3')
        axs[2, 1].plot(np.sqrt(pp.K[:, 1]), '--', c='C3')
        axs[2, 1].plot(np.sqrt(xs_smth))
        axs[0, 2].plot(t_intp, qd_intp[:, [1, 2]])
        axs[0, 2].plot(t_intp, qd_intp, alpha=0.2)
        axs[1, 2].plot(t_intp, qdd_intp[:, [1, 2]])
        axs[1, 2].plot(t_intp, qdd_intp, alpha=0.2)
        axs[2, 2].plot(np.sqrt(pp_intp.K[:, 0]), '--', c='C3')
        axs[2, 2].plot(np.sqrt(pp_intp.K[:, 1]), '--', c='C3')
        axs[2, 2].plot(np.sqrt(xs_intp))
        axs[0, 0].set_title('(colloc.) velocity')
        axs[1, 0].set_title('(colloc.) acceleration')
        axs[2, 0].set_title('(colloc.) profile')
        axs[0, 1].set_title('(smth colloc.) velocity')
        axs[1, 1].set_title('(smth colloc.) acceleration')
        axs[2, 1].set_title('(smth colloc.) profile')
        axs[0, 2].set_title('(intp.) velocity')
        axs[1, 2].set_title('(intp.) acceleration')
        axs[2, 2].set_title('(intp.) profile')
        plt.tight_layout()
        plt.show()

    import IPython
    if IPython.get_ipython() is None:
        IPython.embed()
