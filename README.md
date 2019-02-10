# TOPP-RA
[![Build Status](https://travis-ci.org/hungpham2511/toppra.svg?branch=master)](https://travis-ci.org/hungpham2511/toppra) [![Coverage Status](https://coveralls.io/repos/github/hungpham2511/toppra/badge.svg?branch=master)](https://coveralls.io/github/hungpham2511/toppra?branch=master)


**Documentation and tutorials** are available at (https://hungpham2511.github.io/toppra/).

TOPP-RA is a library for computing the time-optimal path parametrization for robots subject to kinematic and dynamic constraints. 
In general, given the inputs:

1. a geometric path `p(s)`, `s` in `[0, s_end]` ;
2. a list of constraints on joint velocity, joint accelerations, tool Cartesian velocity, et cetera.

TOPP-RA returns the time-optimal path parameterization: `s_dot (s)`, from which the fastest trajectory `q(t)` that satisfies the given
constraints can be found.

## Citing TOPP-RA
If you use this library for your research, we encourage you to 

1. reference the accompanying paper [«A new approach to Time-Optimal Path Parameterization based on Reachability Analysis»](https://www.researchgate.net/publication/318671280_A_New_Approach_to_Time-Optimal_Path_Parameterization_Based_on_Reachability_Analysis), *IEEE Transactions on Robotics*, vol. 34(3), pp. 645–659, 2018.
2. put a star on this repository.


## Bug reports and supports
Please report any issues, questions via [Github issues tracker](https://github.com/hungpham2511/toppra/issues).

It will be very helpful if you can provide more details on the
errors/bugs that you encounter. In fact, the best way is to provide a
Minimal Working Example that produces the reported bug and attach it
with the issue report.

## Contributions

Pull Requests are welcomed! Go ahead and create a Pull Request and we will review your proposal! For new features, or bug fixes, preferably the request should contain unit tests. Note that `toppra` uses [pytest](https://docs.pytest.org/en/latest/contents.html) for all tests. Check out the test folder for more details.
