# TOPP-RA
[![Build Status](https://travis-ci.org/hungpham2511/toppra.svg?branch=master)](https://travis-ci.org/hungpham2511/toppra) [![Coverage Status](https://coveralls.io/repos/github/hungpham2511/toppra/badge.svg?branch=master)](https://coveralls.io/github/hungpham2511/toppra?branch=master)


**Documentation and tutorials** are available at (https://hungpham2511.github.io/toppra/).

TOPP-RA is a library for time-parameterizing robot trajectories subject to kinematic and dynamic constraints. 
In general, given the inputs:

1. a geometric path `q(s)`, `s` in `[0, s_end]` ;
2. a list of constraints on joint velocity, joint accelerations, tool Cartesian velocity, et cetera.

TOPP-RA returns the time-parameterization: `s_dot (s)`, from which a trajectory `q(t)` that satisfies the given
constraints can be computed.

## Citing TOPP-RA
If you use this library for your research, we encourage you to 

1. reference the accompanying paper [«A new approach to Time-Optimal Path Parameterization based on Reachability Analysis»](https://arxiv.org/abs/1707.07239), *IEEE Transactions on Robotics*, vol. 34(3), pp. 645–659, 2018.
2. put a star on this repository.


## Bug reports and supports
Please report any issues, questions via [Github issues tracker](https://github.com/hungpham2511/toppra/issues).

