"""This example shows how to construct a custom PathConstraint from
analytical equations. In this example we want to parametrize a path
s \in [0, 1] subject to the following constraint:

   - (0.5 - s) u + x - 1 <= 0
               u     - 1 <= 0
             - u     - 1 <= 0
                   x     >= 0
"""
import numpy as np
import matplotlib.pyplot as plt

import toppra as ta

# Option
PLOT_PROFILE = True
PLOT_CROSS_SECTION = True
INTERPOLATED = False  # Set of False to use collocation
N = 50

ds = 1. / N

# Constraint
a = np.zeros((N + 1, 4))
b = np.zeros((N + 1, 4))
c = np.zeros((N + 1, 4))
for i, s in enumerate(np.linspace(0, 1, N + 1)):
    a[i] = [- (0.5 - s), 1, -1, 0]
    b[i] = [3, 0, 0, -1]
    c[i] = [-1, -1, -1, 0]
path_constraint = ta.PathConstraint(a=a, b=b, c=c,
                                    ss=np.linspace(0, 1, N + 1))
if INTERPOLATED:
    path_constraint = ta.interpolate_constraint(path_constraint)

# solver
solver = ta.qpOASESPPSolver([path_constraint])
solver.solve_controllable_sets()
Ks = np.copy(solver.K)
# solver.set_start_interval([0.4, 0.42])
# us, xs = solver.solve_topp()

solver.set_start_interval([0.2, 0.22])
us2, xs2 = solver.solve_topp()

# Compute MVC
MVC = []
solver._reset_operational_matrices()
for i in range(N + 1):
    xmin, xmax = solver.proj_x_admissible(i, -10, 10, init=True)
    MVC.append(xmax)
MVC = np.array(MVC)

if PLOT_PROFILE:
    # Normal plot
    fig, axs = plt.subplots(1, 1)
    plot_MVC = axs.plot(MVC, '--', lw=5, c='C0')
    plot_Ks = []
    plot_Ks.extend(axs.plot(Ks[:, 0], lw=5, c='C1'))
    plot_Ks.extend(axs.plot(Ks[:, 1], lw=5, c='C2'))
    # plot_xs = axs.plot(xs, lw=2.5, c='C2')
    plot_xs2 = axs.plot(xs2, lw=2.5, c='C3')
    axs.legend(plot_Ks + plot_xs2 + plot_MVC,
               ['K_min(s)', 'K_max(s)', 'x(s)', 'MVC(s)'])

plt.show()

import IPython
if IPython.get_ipython() is None:
    IPython.embed()
