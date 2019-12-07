Frequently Asked Questions
======================================


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

By default, `toppra` select 100 gridpoints.

As a general rule of thumb, the number of gridpoints should be at
least twice the number of waypoints in the given path.


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

