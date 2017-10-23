"""In this script the use of variable step size to reduce the small
jittering is demonstrated.

In addition, this script also shows how can one quickly generate new
constraint using sympy. It is very convenient!

Finally, this script also compares the result obtained using TOPPRA
with the optimal solution found using cvxpy.
"""
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import sympy

import cvxpy as cvx

import toppra as ta

import cdd

import coloredlogs
coloredlogs.install(level='DEBUG')
np.set_printoptions(3)

# Option
PLOT_PROFILE = True
PLOT_CROSS_SECTION = True
INTERPOLATED = False  # Set of False to use collocation
N = 100


def evaluate_constraint(expressions, ss):
    """Evaluate the canonical constraints `expressions` over the
    discretized interval `ss`.

    Parameters
    ----------
    
    Returns
    -------

    """
    m = len(expressions)
    N = len(ss) - 1
    a = np.zeros((N + 1, m))
    b = np.zeros((N + 1, m))
    c = np.zeros((N + 1, m))
    for index, e in enumerate(expressions):
        if e.rel_op == "<=":
            canon_form = e.lhs - e.rhs
        elif e.rel_op == ">=":
            canon_form = e.rhs - e.lhs
        else:
            raise ValueError("Relation Operation need to be `<=` or `>=`.")

        f = sympy.lambdify(sympy.symbols('u, x, s'), canon_form)
        for i in xrange(N + 1):
            a[i, index] = f(1, 0, ss[i]) - f(0, 0, ss[i])
            b[i, index] = f(0, 1, ss[i]) - f(0, 0, ss[i])
            c[i, index] = f(0, 0, ss[i])
    return a, b, c


# Define symbolic expression and evaluation
ss = np.linspace(0, 0.495, N / 4)
ss = np.hstack((ss, np.linspace(0.496, 0.51, N / 2)))
ss = np.hstack((ss, np.linspace(0.511, 1, N / 4)))
N = ss.shape[0] - 1

# ss = np.linspace(0, 1, N + 1)

u, x, s = sympy.symbols('u x s')
expressions = [
    - 0.2 * (0.5 - s) * u + 3 * x - 1 <= 0,
    x >= 0,
    u <= 1, u >= -1
]
a, b, c = evaluate_constraint(expressions, ss)

path_constraint = ta.PathConstraint(a=a, b=b, c=c, ss=ss)
if INTERPOLATED:
    path_constraint = ta.interpolate_constraint(path_constraint)

# Solve with TOPPRA
solver = ta.qpOASESPPSolver([path_constraint])
solver.solve_controllable_sets()
Ks = np.copy(solver.K)
solver.set_start_interval([0.1, 0.12])
us, xs = solver.solve_topp()

solver.set_start_interval([0.2, 0.22])
us2, xs2 = solver.solve_topp()

# Compute MVC
MVC = []
solver._reset_operational_matrices()
for i in range(N + 1):
    xmin, xmax = solver.proj_x_admissible(i, -10, 10, init=True)
    MVC.append(xmax)

# Solve with cvxpy
x_var = cvx.Variable(N + 1)
u_var = cvx.Variable(N)

constraints = []
obj = 0
for i in range(N):
    dsi = ss[i + 1] - ss[i]
    # Continuity constraints
    constraints.append(x_var[i] + 2 * dsi * u_var[i] == x_var[i + 1])

    # Path constraints
    constraints.append(u_var[i] * path_constraint.a[i]
                       + x_var[i] * path_constraint.b[i]
                       + path_constraint.c[i] <= 0)

    obj += 2 * dsi * cvx.power(cvx.sqrt(x_var[i]) + cvx.sqrt(x_var[i + 1]), -1)

constraints.append(x_var[0] == 0.2)
constraints.append(x_var[N] == 0)
obj = cvx.Minimize(obj)

prob = cvx.Problem(obj, constraints)
prob.solve()
xs_cvx = np.array(x_var.value).flatten()

cm = {
    'K': 'C0',
    'p1': 'C1', 'p2': 'C2', 'p3': 'C3',
    'MVC': 'C4'}

if PLOT_PROFILE:
    # Normal plot
    fig, axs = plt.subplots(1, 1)
    plots = []
    plot_MVC = axs.plot(ss, MVC, '--', lw=5, c=cm['MVC'])
    plot_Ks = axs.plot(ss, Ks, lw=5, c=cm['K'])
    plot_xs = axs.plot(ss, xs, lw=2.5, c=cm['p1'])
    plot_xs2 = axs.plot(ss, xs2, lw=2.5, c=cm['p3'])
    plot_xscvx = axs.plot(ss, xs_cvx, lw=2.5, c=cm['p2'])
    axs.legend(plot_Ks + plot_xs + plot_xs2 + plot_xscvx + plot_MVC,
               ['K_min', 'K_max',
                'x1 (comp. TOPPRA)',
                'x2 (comp. TOPPRA)',
                'x3 (comp. CVXPY)',
                'MVC'])

plt.show()

import IPython
if IPython.get_ipython() is None:
    IPython.embed()
