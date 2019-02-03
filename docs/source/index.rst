.. TOPP-RA documentation master file, created by
   sphinx-quick-start on Sat Nov 18 00:03:54 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

TOPP-RA: Fast path-parameterization for robots
===================================================

TOPP-RA is a library for computing `path parameterizations` of
arbitrary geometric paths/trajectories for robots, which can be either
industrial robot manipulators or humanoids!  TOPP-RA can account for
several types of constraints on robot dynamics:

1. joint torque, velocity and acceleration bounds;
2. *robust* joint torque, velocity and acceleration bounds;
3. Cartesian acceleration bounds;
4. contact stability for legged robots.
5. Your constraint! See the tutorials to understand how to implement
   your own constraints and handle them with TOPP-RA.

You can use TOPP-RA to compute the *time-optimal* path parameterization
-- the fastest possible movement the robot can realize -- or a path
parameterization with *a specified duration*. See the tutorials for
more details.

TOPP-RA is efficient. For standard use cases, it should return a
solution in 5ms-10ms.
  
See below for some tutorials, modules reference and installation
instructions.

.. toctree::
   :maxdepth: 2

   tutorials
   modules
   installation

.. _path-parameterization:
 

Citing TOPP-RA!
~~~~~~~~~~~~~~~~
If you find TOPP-RA useful and use it in your research, we encourage
you to

1. reference the accompanying paper `«A new approach to Time-Optimal Path Parameterization based on Reachability Analysis» <https://arxiv.org/abs/1707.07239>`_ *IEEE Transactions on Robotics*, vol. 34(3), pp. 645–659, 2018.
2. put a star on this repository!


Bug reports and supports
~~~~~~~~~~~~~~~~~~~~~~~~~
Please report any issues, questions via `Github issues tracker <https://github.com/hungpham2511/toppra/issues>`_.
