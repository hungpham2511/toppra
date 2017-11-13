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

import coloredlogs
coloredlogs.install(level='INFO')
np.set_printoptions(3)

# Option
PLOT_PROFILE = True
PLOT_CROSS_SECTION = True
INTERPOLATED = True  # Set of False to use collocation
N = 40

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
solver.set_start_interval([0, 0])
solver.solve_reachable_sets()
Ls = np.copy(solver.L)  # Reachable sets
# The propagated velocity interval can be retrieved as Ls[-1]

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
    plot_Ls = []
    plot_Ls.extend(axs.plot(Ls[:, 0], lw=5, c='C2'))
    axs.plot(Ls[:, 1], lw=5, c='C2')
    # plot_xs = axs.plot(xs, lw=2.5, c='C2')
    axs.legend(plot_Ls + plot_MVC,
               ['Reachable sets', 'MVC'])
plt.show()

import IPython
if IPython.get_ipython() is None:
    IPython.embed()
