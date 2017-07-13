import numpy as np
from toppra import (qpOASESPPSolver,
                    create_velocity_path_constraint,
                    create_full_contact_path_constraint,
                    paper_dir,
                    interpolate_constraint,
                    compute_trajectory_gridpoints)

import matplotlib.pyplot as plt
from rave.Rave import (normalize, UnivariateSplineInterpolator)

import pymanoid
from pymanoid import PointMass, Stance, Contact
import openravepy as orpy

com_height = 0.9  # [m]
z_polygon = 2.
VERBOSE = False
N = 20
SAVE_FIG = False


def load_problem_data(load_viewer=True):
    """ Load pymanoid code and generate the path.
    """
    class COMSync(pymanoid.Process):

        def on_tick(self, sim):
            com_above.set_x(com_target.x)
            com_above.set_y(com_target.y)

    sim = pymanoid.Simulation(dt=0.03)
    robot = pymanoid.robots.JVRC1('JVRC-1.dae', download_if_needed=True)
    sim.BACKGROUND_COLOR = np.r_[230., 230, 231] / 255
    if load_viewer:
        sim.set_viewer()
        sim.viewer.SetCamera([
            [0.60587192, -0.36596244,  0.70639274, -2.4904027],
            [-0.79126787, -0.36933163,  0.48732874, -1.6965636],
            [0.08254916, -0.85420468, -0.51334199,  2.79584694],
            [0.,  0.,  0.,  1.]])
    if not VERBOSE:
        orpy.RaveSetDebugLevel(orpy.DebugLevel.Fatal)

    robot.set_transparency(0.10)
    robot.set_dof_values([
        3.53863816e-02,   2.57657518e-02,   7.75586039e-02,
        6.35909636e-01,   7.38580762e-02,  -5.34226902e-01,
        -7.91656626e-01,   1.64846093e-01,  -2.13252247e-01,
        1.12500819e+00,  -1.91496369e-01,  -2.06646315e-01,
        1.39579597e-01,  -1.33333598e-01,  -8.72664626e-01,
        0.00000000e+00,  -9.81307787e-15,   0.00000000e+00,
        -8.66484961e-02,  -1.78097540e-01,  -1.68940240e-03,
        -5.31698601e-01,  -1.00166891e-04,  -6.74394930e-04,
        -1.01552628e-04,  -5.71121132e-15,  -4.18037117e-15,
        0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
        0.00000000e+00,  -7.06534763e-01,   1.67723830e-01,
        2.40289101e-01,  -1.11674923e+00,   6.23384177e-01,
        -8.45611535e-01,   1.39994759e-02,   1.17756934e-16,
        3.14018492e-16,  -3.17943723e-15,  -6.28036983e-16,
        -3.17943723e-15,  -6.28036983e-16,  -6.88979202e-02,
        -4.90099381e-02,   8.17415141e-01,  -8.71841480e-02,
        -1.36966665e-01,  -4.26226421e-02])

    com_target = PointMass(
        pos=[0., 0., com_height], mass=robot.mass, color='b', visible=False)
    com_above = pymanoid.Cube(0.02, [0.05, 0.04, z_polygon], color='b')

    left_ankle = robot.rave.GetLink('L_ANKLE_P_S')
    right_ankle = robot.rave.GetLink('R_ANKLE_P_S')

    stance = Stance(
        com=com_target,
        left_foot=Contact(
            shape=robot.sole_shape,
            pos=[0.20, 0.15, 0.1],
            rpy=[0.4, 0, 0],
            friction=0.5,
            visible=True,
            link=left_ankle),
        right_foot=Contact(
            shape=robot.sole_shape,
            pos=[-0.2, -0.195, 0.],
            rpy=[-0.4, 0, 0],
            friction=0.5,
            visible=True,
            link=right_ankle))
    stance.bind(robot)
    robot.ik.solve()

    # Generate a simple full-body path by tracking the COM path.
    try:
        ts, qs, _ = np.load("_temp_{}.npy".format(__name__))
    except:
        com_sync = COMSync()
        sim.schedule(robot.ik)
        sim.schedule_extra(com_sync)
        com_target.set_x(0)
        com_target.set_y(0)
        robot.ik.solve()
        ts = np.arange(0, 7, sim.dt)
        ps = np.array([com_target.p +
                       np.r_[0.1, 0.1, 0] * t for t in np.sin(ts)])
        qs = []
        for p in ps:
            qs.append(robot.q)
            com_target.set_pos(p)
            sim.step(1)
        sim.stop()
        qs = np.array(qs)
        np.save("_temp_{}.npy".format(__name__), [ts, qs, None])
        print "Does not found stored path, generated and saved new one."

    # Interpolation
    path = UnivariateSplineInterpolator(normalize(ts), qs)

    # Set Torque bounds
    tau_bnd = np.ones(robot.nb_dofs) * 100.
    tau_bnd[[robot.TRANS_X, robot.TRANS_Y, robot.TRANS_Z,
             robot.ROT_R, robot.ROT_P, robot.ROT_Y]] = 0.
    robot.rave.SetDOFTorqueLimits(tau_bnd)
    return path, robot, stance


def smooth_singularities(pp, us, xs, vs=None):
    """Smooth jitters due to singularities.

    Solving TOPP for discrete problem generated from collocation
    scheme tends to create jitters. This function finds and smooth
    them.

    Args:
    ----
    pp: PathParameterization
    us: ndarray
    xs: ndarray
    vs: ndarray, optional

    Returns:
    -------
    us_smth: ndarray,
    xs_smth: ndarray,
    vs_smth: ndarray,
    """
    # Find the indices
    singular_indices = []
    uds = np.diff(us, n=1)
    for i in range(pp.N - 3):
        if uds[i] < 0 and uds[i+1] > 0 and uds[i+2] < 0:
            print "Potential peak at {:d}".format(i)
            singular_indices.append(i)
    print "Found singularities at {}".format(singular_indices)

    # Smooth the singularities
    xs_smth = np.copy(xs)
    us_smth = np.copy(us)
    if vs is not None:
        vs_smth = np.copy(vs)
    for index in singular_indices:
        idstart = max(0, index)
        idend = min(pp.N, index + 4)
        xs_smth[range(idstart, idend + 1)] = (
            xs_smth[idstart] + (xs_smth[idend] - xs_smth[idstart]) *
            np.linspace(0, 1, idend + 1 - idstart))
        if vs is not None:
            data = [vs_smth[idstart] +
                    (xs_smth[idend] - xs_smth[idstart]) * frac
                    for frac in np.linspace(0, 1, idend + 1 - idstart)]
            vs_smth[range(idstart, idend + 1)] = np.array(data)

    for i in range(pp.N):
        us_smth[i] = (xs_smth[i+1] - xs_smth[i]) / 2 / (pp.ss[i+1] - pp.ss[i])

    if vs is not None:
        return us_smth, xs_smth, vs_smth
    else:
        return us_smth, xs_smth


if __name__ == "__main__":
    path, robot, stance = load_problem_data()
    N = 100
    ss = np.linspace(0, 1, N+1)
    pc_full_contact = create_full_contact_path_constraint(
        path, ss, robot, stance)
    vlim_ = np.ones(robot.nb_dofs)
    vlim = np.vstack((-vlim_, vlim_)).T
    pc_vel = create_velocity_path_constraint(path, ss, vlim)

    # collocation
    pcs = [pc_vel, pc_full_contact]
    pp = qpOASESPPSolver(pcs)
    us, xs = pp.solve_topp(reg=1e-6, save_solutions=True)
    vs = pp.slack_vars
    us, xs, vs = smooth_singularities(pp, us, xs, vs)
    t, q, qd, qdd = compute_trajectory_gridpoints(path, ss, us, xs)

    # interpolation
    pp_intp = qpOASESPPSolver([interpolate_constraint(pc) for pc in pcs])
    us_intp, xs_intp = pp_intp.solve_topp(reg=1e-6, save_solutions=True)
    t_intp, q_intp, qd_intp, qdd_intp = compute_trajectory_gridpoints(
        path, ss, us_intp, xs_intp)
    vs_intp = pp_intp.slack_vars

    # plotting
    plt.rcParams['axes.labelsize'] = 'small'
    plt.rcParams['legend.fontsize'] = 6
    plt.rcParams['xtick.labelsize'] = 6
    plt.rcParams['ytick.labelsize'] = 6
    f, axs = plt.subplots(2, 1, figsize=[2.8, 2.6])
    axs[0].plot(pp.ss, np.sqrt(pp.K[:, 0]), '--', c='C3')
    axs[0].plot(pp.ss, np.sqrt(pp.K[:, 1]), '--', c='C3')
    axs[0].plot(pp.ss, np.sqrt(xs), "-")
    axs[1].plot(t[:-1], vs[:, 50], label='$F_x$')
    axs[1].plot(t[:-1], vs[:, 51], label='$F_y$')
    axs[1].plot(t[:-1], vs[:, 52], label='$F_z$')
    axs[1].plot(t[:-1], vs[:, [56, 57, 58]], '--', alpha=0.5)
    plt.legend(loc='upper right')
    plt.tight_layout()
    if SAVE_FIG:
        plt.savefig("{}/humanoid_swaying.pdf".format(paper_dir()))
    else:
        axs[0].set_title("Phase plane")
        axs[1].set_title("Contact forces/Joint torques")
        plt.show()

    # import time
    # for i in range(100):
    #     robot.set_dof_values(q[i])
    #     print "{:d} at t={:8.3f}".format(i, t[i])
    #     time.sleep(3 * (t[i+1] - t[i]))

    import IPython
    if IPython.get_ipython() is None:
        IPython.embed()
