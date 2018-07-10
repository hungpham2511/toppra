# TOPP-RA
[![Build Status](https://travis-ci.org/hungpham2511/toppra.svg?branch=master)](https://travis-ci.org/hungpham2511/toppra) [![Coverage Status](https://coveralls.io/repos/github/hungpham2511/toppra/badge.svg?branch=master)](https://coveralls.io/github/hungpham2511/toppra?branch=master)

TOPP-RA is a library for time-parameterizing robot trajectories subject to kinematic and dynamic constraints. The current implementation supports the following constraints:

1. joint velocity and acceleration bounds;
2. torque bounds (including redundantly-actuated manipulators);
3. contact stability for legged robots.

If you use this library for your research, please reference the accompanying paper [«A new approach to Time-Optimal Path Parameterization based on Reachability Analysis»](https://arxiv.org/abs/1707.07239), *IEEE Transactions on Robotics*, vol. 34(3), pp. 645–659, 2018.

# Installation
## Basic functionality (robotic manipulators)


Install
[qpOASES](https://projects.coin-or.org/qpOASES/wiki/QpoasesInstallation) by
following the steps below:
``` shell
git clone https://github.com/hungpham2511/qpOASES
cd qpOASES/ && mkdir bin && make
cd interfaces/python/
pip install cython
python setup.py install --user
```

Finally, install `toppra` with
``` sh
git clone https://github.com/hungpham2511/toppra
cd toppra/
pip install -r requirements.txt --user
pip install -e . --user
```
And you are good to go. If you have `openrave` installed on your computer, you can
run the below example to see `toppra` in action.

``` shell
python examples/retime_rave_trajectory.py
```

## Advanced (and unstable) functionality 

Multi-contact and torque bounds.  To use these functionality, the
following libraries are needed:

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
git checkout 54299cf
export PYTHONPATH=$PYTHONPATH:$HOME/git/pymanoid
```

## Building docs
To build and view the documentation, install
[sphinx](http://www.sphinx-doc.org/en/stable/index.html) then do
``` shell
pip install sphinx_rtd_theme
cd <toppra-dir>/docs/
make clean && make html
<google-chrome> build/index.html
```

## Test
`toppra` uses `pytest` for testing. To run all the tests, do:
``` sh
cd <toppra-dir>/tests/
pytest -v
```
if `pytest` is not installed, grab it from `pip`.
