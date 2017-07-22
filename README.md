# `toppra`

A library for computing path parameterizations for robots subject to
constraints. `toppra` can consider the following constraints on a
robots:

1. joint velocity and acceleration bounds;
2. torque bound;
3. contact stability;
4. full rigid-body dynamics with linearized friction cone constraints.


# Installation
## Basic

From pip

``` sh
pip install Cython numpy coloredlogs enum scipy
```

to run the examples, you need in addition

``` sh
pip install matplotlib
```


Install
[qpOASES](https://projects.coin-or.org/qpOASES/wiki/QpoasesInstallation) by
following steps given in the official site. Then, to install the
python interface do
``` sh
cd <install-dir>/interfaces/python
pip setup.py install
```


Finall, install `toppra` with
``` sh
python setup.py install
```
after installation complete, run tests by doing
``` sh
cd <toppra-dir>/tests/
pytest -v
```
if `pytest` is not installed, grab it from `pip`.



## Multi-contact and torque bounds examples
To use these functionalities, the following libraries are needed:

1. [openRAVE](https://github.com/rdiankov/openrave)
2. [pymanoid](https://github.com/stephane-caron/pymanoid)

`openRAVE` can be tricky to install, a good instruction for installing
`openRAVE` on Ubuntu 16.04 can be
found
[here](https://scaron.info/teaching/installing-openrave-on-ubuntu-16.04.html).

To install `pymanoid` locally, do the following
``` sh
mkdir git && cd git
git clone <pymanoid-git-url>
export PYTHONPATH=$PYTHONPATH:$HOME/git/pymanoid
```


# Basic usage

The following shows basic usages of `toppra`. First, import necessary
functions
```python
from toppra import (create_velocity_path_constraint,
                    create_acceleration_path_constraint,
                    qpOASESPPSolver,
                    compute_trajectory_gridpoints,
                    smooth_singularities,
		    SplineInterpolator,
                    interpolate_constraint)
import numpy as np
```
Then, we generate a random instance.
```python
way_pts = np.random.randn(N_samples, n)
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
```
And finally solve with `toppra`.
```python
pp = qpOASESSolver(constraints)
us, xs = pp.solve_topp()
us, xs = smooth_singularities(pp, us, xs)
t, q, qd, qdd = compute_trajectory_gridpoints(path, pp.ss, us, xs)
```



