.. _notes:

Frequently Asked Questions
================================


1. How many gridpoints should I take?
---------------------------------------

A very important parameter in solving path parameterization instances
with `toppra` is the number of the gridpoints. Below is how to create
an instance with 1000 uniform gridpoints.


.. code-block:: python
  :linenos:

  gridpoints = np.linspace(0, path.duration, 1000)  # 1000 points
  instance = algo.TOPPRA([pc_vel, pc_acc], path, gridpoints=gridpoints)


Generally, more gridpoints give you better solution quality, but also
increase solution time. Often the increase in solution time is linear,
that is if it takes 5ms to solve an instance with 100 gridpoints, then
most likely `toppra` will take 10ms to solve another instance which
has 200 gridpoints.

As a general rule of thumb, the number of gridpoints should be at
least a few times the number of waypoints in the given path. This is
not a hard rule, depending on whether the waypoints naturally form a
smooth curve or whether they vary wildly.

By default, `toppra` (python) will try to determine the best set of
gridpoints by doing a bisection search until a threshold level is
reached.


2. Minimum requirement on path smoothness
-------------------------------------------------

TOPPRA requires the input path to be sufficiently smooth to work
properly. An example of a noisy path that will be very difficult to
work with can be seen below:

.. image:: medias/faq_figure.png

All toppra interpolators try to match all given waypoints, and hence
it can lead to large fluctuation if the waypoints change rapidly. In
this case, it is recommended to smooth the waypoints prior to using
toppra using for example `scipy.interpolation`.


Usage Notes
=====================

.. _derivationKinematics:

Derivation of kinematical quantities
------------------------------------

In `toppra` we deal with geometric paths, which are mathematically
functions :math:`\mathbf p(s)`. Here :math:`s` is the path position
and usually belongs to the interval :math:`[0, 1]`. Notice that
`toppra` can also handle arbitrary interval. In the code a path is
represented by a child class inherited from the abstract
:class:`toppra.interpolator.AbstractGeometricPath`.


Important expression relating kinematic quantities:

.. math::
   \mathbf q(t) &= \mathbf p(s(t)) \\
   \dot{\mathbf p}(t) &= \mathbf p'(s) \dot s(t) \\
   \ddot{\mathbf p}(t) &= \mathbf p'(s) \ddot s(t) + \mathbf p''(s) \dot s(t)^2

