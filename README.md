# toppra

A library for computing path parameterizations for robots subject to
constraints. The library can consider the following constraints:

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


Install qpOASES, the numerical solver

``` sh
git clone https://github.com/stephane-caron/qpOASES
cd qpOASES
```


Finall, install `toppra` with

``` sh
sudo python setup.py install
```

## Multi-contact and torque bounds examples
To use these functionalities, the following libraries are needed:

1. [openRAVE](https://github.com/rdiankov/openrave)
2. [pymanoid](https://github.com/stephane-caron/pymanoid)

`openRAVE` can be tricky to install, a good instruction for installing
`openRAVE` on Ubuntu 16.06 can be
found
[here](https://scaron.info/teaching/installing-openrave-on-ubuntu-16.04.html).


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



