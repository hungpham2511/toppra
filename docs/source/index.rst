:code:`toppra` : Path-parameterization for robots
===================================================

|github release| |docs| |circleci|

|pypi|

.. |pypi| image:: https://badge.fury.io/py/toppra.svg
    :target: https://badge.fury.io/py/toppra

.. |docs| image:: https://readthedocs.org/projects/toppra/badge/?version=latest
    :target: https://toppra.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. |github release| image:: https://img.shields.io/github/release/hungpham2511/toppra
   :target: https://github.com/hungpham2511/toppra/releases/

.. |circleci| image:: https://circleci.com/gh/hungpham2511/toppra/tree/develop.svg?style=svg
    :target: https://circleci.com/gh/hungpham2511/toppra/tree/develop



TOPP-RA is a library for computing path parametrizations for geometric
paths subject to certain forms of kinematic and dynamic constraints
Simply speaking given

1. a smooth geometric path :math:`p(s), s \in [0, s_{end}]` ;
2. a list of constraints on joint velocity, joint accelerations, tool
   Cartesian velocity, et cetera.

TOPP-RA can produce the time-optimal path parameterization
:math:`s_{dot} (s)`, from which the fastest trajectory `q(t)` that
satisfies the given constraints can be found. All of this is done in a
few milliseconds.  There are some additional features that make it
easier to generate high-quality motions for robots:

1. Return the time-optimal parametrization or a parametrization with
   specified duration subject to constraints.
2. Able to handle multiple constraint types:
3. Automatic grid-points selection.

.. figure:: _static/toppra_illus.png
	    
   This is the time-optimal joint trajectory for a robot subject to
   joint accelerations and velocities constraints computed using
   TOPP-RA. See :ref:`tutorials` for more examples.
  
Site overview
----------------------

.. toctree::
   :maxdepth: 1

   installation
   tutorials
   notes
   python_api
   HISTORY


Citing TOPP-RA!
----------------
If you find TOPP-RA useful and use it in your research, we encourage
you to

1. reference the accompanying paper `«A new approach to Time-Optimal Path Parameterization based on Reachability Analysis» <https://www.researchgate.net/publication/318671280_A_New_Approach_to_Time-Optimal_Path_Parameterization_Based_on_Reachability_Analysis>`_ *IEEE Transactions on Robotics*, vol. 34(3), pp. 645–659, 2018.
2. put a star on this repository!


Bug reports and supports
-------------------------
Please report any issues, questions via `Github issues tracker <https://github.com/hungpham2511/toppra/issues>`_.


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


