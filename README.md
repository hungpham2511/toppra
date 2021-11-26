# `toppra`: Time-Optimal Path Parameterization
![Integrate](https://github.com/hungpham2511/toppra/actions/workflows/integrate.yml/badge.svg)


- [Overview](#overview)
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

**Documentation and tutorials** are available
[here](https://hungpham2511.github.io/toppra/index.html).

You can install the package with pip:

``` shell
pip install toppra
```

To install from source for development:

``` shell
pip install -r requirement3.txt
pip install -e .
```

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
Group) and [Phạm Quang Cưong](https://personal.ntu.edu.sg/cuong/)
(Eureka Robotics, CRI Group) with major contributions from talented
contributors:
- [Joseph Mirabel](https://github.com/jmirabel) (C++ API)
- EdsterG (Python3 support).

If you have taken part in developing and supporting the library, feel
free to add your name to the list.

The development is also generously supported by [Eureka Robotics](https://eurekarobotics.com/).

# Citing toppra
If you use this library for your research, we encourage you to 

1. reference the accompanying paper [A new approach to Time-Optimal Path Parameterization based on Reachability Analysis](https://www.researchgate.net/publication/318671280_A_New_Approach_to_Time-Optimal_Path_Parameterization_Based_on_Reachability_Analysis),
   *IEEE Transactions on Robotics*, vol. 34(3), pp. 645-659, 2018.
2. put a star on this repository.

