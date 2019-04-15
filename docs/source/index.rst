.. TOPP-RA documentation master file, created by
   sphinx-quick-start on Sat Nov 18 00:03:54 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

TOPP-RA: Fast path-parameterization for robots
===================================================

TOPP-RA is a library for computing `path parameterizations` of
arbitrary geometric paths/trajectories for robots!

TOPP-RA can account for several types of constraints on robot
dynamics:

1. joint torque, velocity and acceleration bounds;
2. *robust* joint torque, velocity and acceleration bounds;
3. Cartesian acceleration bounds;
4. contact stability for legged robots.
5. Your constraint! See the tutorials to understand how to implement
   your own constraints and handle them with TOPP-RA.

You can use TOPP-RA to compute the *time-optimal* path
parameterization -- the fastest possible movement the robot can
realize -- or a path parameterization with *a specified duration*. See
the tutorials for more details.

A last remark, TOPP-RA is efficient. For standard use cases, it should
return a solution in 5ms-10ms.

**Update (Feb, 2019)** I have used TOPP-RA to plan *critically fast*
motions for robots doing bin picking with suction cup. Here
*critically fast* motions are those that are fastest possible given
the limited suction power and object weight. See the video below for
more detail!

.. raw:: html

   <iframe width="560" height="315" src="https://www.youtube.com/embed/b9H-zOYWLbY" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>


If you find this interesting, feel free to check out the paper: `«Critically fast pick-and-place with suction cups» <https://www.researchgate.net/publication/327570258_Critically_fast_pick-and-place_with_suction_cups>`_. This paper will be presented at ICRA 2019. Feel free to come over to our poster for a chat!
  
You can find on this page :ref:`installation`, :ref:`tutorials`, some
:ref:`notes` and :ref:`module_ref`.

.. toctree::
   :hidden:
   :maxdepth: 3

   installation
   tutorials
   notes
   modules

Citing TOPP-RA!
~~~~~~~~~~~~~~~~
If you find TOPP-RA useful and use it in your research, we encourage
you to

1. reference the accompanying paper `«A new approach to Time-Optimal Path Parameterization based on Reachability Analysis» <https://www.researchgate.net/publication/318671280_A_New_Approach_to_Time-Optimal_Path_Parameterization_Based_on_Reachability_Analysis>`_ *IEEE Transactions on Robotics*, vol. 34(3), pp. 645–659, 2018.
2. put a star on this repository!


Bug reports and supports
~~~~~~~~~~~~~~~~~~~~~~~~~
Please report any issues, questions via `Github issues tracker <https://github.com/hungpham2511/toppra/issues>`_.
