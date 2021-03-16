# `toppra`: Time-Optimal Path Parameterization
[![CircleCI](https://circleci.com/gh/hungpham2511/toppra/tree/develop.svg?style=svg)](https://circleci.com/gh/hungpham2511/toppra/tree/develop)
[![Coverage Status](https://coveralls.io/repos/github/hungpham2511/toppra/badge.svg?branch=master)](https://coveralls.io/github/hungpham2511/toppra?branch=master)
[![Documentation Status](https://readthedocs.org/projects/toppra/badge/?version=latest)](https://toppra.readthedocs.io/en/latest/?badge=latest)


- [Overview](#overview)
- [Supports](#supports)
- [Slack](https://join.slack.com/t/toppracommunity/shared_invite/zt-npyxpszp-3msp6MZhE5wNcjdrB~fWFg)
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

**Documentation and tutorials** are available
[here](https://hungpham2511.github.io/toppra/index.html).

**Development roadmap** Please check our the Issues tracker.

# Support

## Bug tracking
Please report any issues, questions or feature request via 
[Github issues tracker](https://github.com/hungpham2511/toppra/issues).

Have a quick question? Try asking in our slack channel.

## Contributions
Pull Requests are welcome! Create a Pull Request and we will review
your proposal!

# Credits

`toppra` was originally developed by [Hung
Pham](https://hungpham2511.github.com/) (Eureka Robotics, former CRI
Group) and [Phạm Quang Cường](https://www.ntu.edu.sg/home/cuong/)
(Eureka Robotics, CRI Group) with major contributions from talented
contributors:
- [Joseph Mirabel](https://github.com/jmirabel) (C++ API)
- EdsterG (Python3 support).

If you have taken part in developing and supporting the library, feel
free to add your name to the list.

The development is also generously supported by [Eureka Robotics](https://eurekarobotics.com/).

# Citing toppra
If you use this library for your research, we encourage you to 

1. reference the accompanying paper [«A new approach to Time-Optimal Path Parameterization based on Reachability Analysis»](https://www.researchgate.net/publication/318671280_A_New_Approach_to_Time-Optimal_Path_Parameterization_Based_on_Reachability_Analysis),
   *IEEE Transactions on Robotics*, vol. 34(3), pp. 645–659, 2018.
2. put a star on this repository.

