# `toppra`
[![CircleCI](https://circleci.com/gh/hungpham2511/toppra/tree/develop.svg?style=svg)](https://circleci.com/gh/hungpham2511/toppra/tree/develop)
[![Coverage Status](https://coveralls.io/repos/github/hungpham2511/toppra/badge.svg?branch=master)](https://coveralls.io/github/hungpham2511/toppra?branch=master)
[![Documentation Status](https://readthedocs.org/projects/toppra/badge/?version=latest)](https://toppra.readthedocs.io/en/latest/?badge=latest)


- [Overview](#overview)
- [Development roadmap](#development-roadmap)
- [Supports](#supports)
- [Citing `toppra`](#citing--toppra-)


# Overview

**toppra** is a library for computing the time-optimal path
parametrization for robots subject to kinematic and dynamic
constraints.  In general, given the inputs:

1. a geometric path `p(s)`, `s` in `[0, s_end]`;
2. a list of constraints on joint velocity, joint accelerations, tool
   Cartesian velocity, et cetera.

**toppra** returns the time-optimal path parameterization: `s_dot
(s)`, from which the fastest trajectory `q(t)` that satisfies the
given constraints can be found.

**Documentation and tutorials** are available at
(https://toppra.readthedocs.io/en/latest/index.html).


## Quick-start

To install **TOPP-RA**, simply clone the repo and install with pip

``` shell
git clone https://github.com/hungpham2511/toppra
cd toppra && pip install .
```

To install depencidencies for development, replace the second command with:
``` shell
cd toppra && pip install -e .[dev]
```

# Development roadmap

The following is a non-exhautive list of features that we are
considering to include in the library.

- Improve the trajectory / geometric path interface [#81](https://github.com/hungpham2511/toppra/issues/81)
- Implement a C++ interface [#43](https://github.com/hungpham2511/toppra/issues/43)
- Implement a C++ interface to popular motion planning libraries.
- Improve the numerical stability of the solvers for degenerate cases.
- Post-processing of output trajectories: [#56](https://github.com/hungpham2511/toppra/issues/56), [#80](https://github.com/hungpham2511/toppra/issues/80)

## Contributions

Pull Requests are welcomed!
- Go ahead and create a Pull Request and we will review your proposal!
- For new features, or bug fixes, preferably the request should
  contain unit tests. Note that `toppra` uses
  [pytest](https://docs.pytest.org/en/latest/contents.html) for all
  tests. Check out the test folder for more details.

# Supports
Please report any issues, questions or feature request via 
[Github issues tracker](https://github.com/hungpham2511/toppra/issues).

Please provide more details on the errors/bugs that you encounter. The
best way is to provide a Minimal Working Example that produces the
reported bug and attach it with the issue report.


# Citing `toppra`
If you use this library for your research, we encourage you to 

1. reference the accompanying paper [«A new approach to Time-Optimal Path Parameterization based on Reachability Analysis»](https://www.researchgate.net/publication/318671280_A_New_Approach_to_Time-Optimal_Path_Parameterization_Based_on_Reachability_Analysis),
   *IEEE Transactions on Robotics*, vol. 34(3), pp. 645–659, 2018.
2. put a star on this repository.

