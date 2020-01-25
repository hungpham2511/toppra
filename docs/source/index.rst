.. TOPP-RA documentation master file, created by
   sphinx-quick-start on Sat Nov 18 00:03:54 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

**TOPP-RA**: Path-parameterization for robots
===================================================


.. image:: https://circleci.com/gh/hungpham2511/toppra/tree/develop.svg?style=svg
    :target: https://circleci.com/gh/hungpham2511/toppra/tree/develop

.. image:: https://readthedocs.org/projects/toppra/badge/?version=latest
    :target: https://toppra.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

**TOPP-RA** is a library for computing the time-optimal path
parametrization for robots subject to kinematic and dynamic
constraints.  In general, given the inputs:

1. a geometric path `p(s)`, `s` in `[0, s_end]` ;
2. a list of constraints on joint velocity, joint accelerations, tool
   Cartesian velocity, et cetera.

**TOPP-RA** returns the time-optimal path parameterization: `s_dot
(s)`, from which the fastest trajectory `q(t)` that satisfies the
given constraints can be found. All of this is done in a few
milliseconds.

Features
---------

1. Return the time-optimal parametrization or a parametrization with
   specified duration subject to constraints.
2. Able to handle multiple constraint types:
  1. joint torque, velocity and acceleration bounds;
  2. *robust* joint torque, velocity and acceleration bounds;
  3. Cartesian acceleration bounds;
  4. contact stability for legged robots.
3. Automatic grid-points selection.

Applications
------------

**(Feb, 2019)** TOPP-RA was used to plan *critically fast*
motions for robots doing bin picking with suction cup. Here
*critically fast* motions are those that are fastest possible given
the limited suction power and object weight. See the video below for
more detail!

.. raw:: html

   <iframe width="560" height="315" src="https://www.youtube.com/embed/b9H-zOYWLbY" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>


If you find this interesting, feel free to check out the paper:
`«Critically fast pick-and-place with suction cups»
<https://www.researchgate.net/publication/327570258_Critically_fast_pick-and-place_with_suction_cups>`_. This
paper has been presented at ICRA 2019.

User Guide
----------
  
You can find on this page :ref:`installation`, :ref:`tutorials`, some
:ref:`notes` and :ref:`module_ref`.

Citing TOPP-RA!
----------------
If you find TOPP-RA useful and use it in your research, we encourage
you to

1. reference the accompanying paper `«A new approach to Time-Optimal Path Parameterization based on Reachability Analysis» <https://www.researchgate.net/publication/318671280_A_New_Approach_to_Time-Optimal_Path_Parameterization_Based_on_Reachability_Analysis>`_ *IEEE Transactions on Robotics*, vol. 34(3), pp. 645–659, 2018.
2. put a star on this repository!


Bug reports and supports
-------------------------
Please report any issues, questions via `Github issues tracker <https://github.com/hungpham2511/toppra/issues>`_.

.. toctree::
   :hidden:
   :maxdepth: 3

   installation
   tutorials
   notes
   FAQs
   modules

