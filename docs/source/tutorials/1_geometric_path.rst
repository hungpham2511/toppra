Path and Trajectory representation
=====================================================

In TOPP-RA, both geometric paths :math:`\mathbf q(s)_{s \in [0, 1]}`
and parametrized trajectories :math:`\mathbf q(t)_{t \in [0, T]}` are
represented by :class:`toppra.Interpolator`. The most important
functions are :func:`~toppra.Interpolator.eval`,
:func:`~toppra.Interpolator.evald` and
:func:`~toppra.Interpolator.evaldd`. These functions evaluate the
configuration, first-derivatives and second-derivatives respectively.


Spline Interpolator
--------------------

In the examples, the child class :class:`toppra.SplineInterpolator` is
used extensively, due to the expressiveness and convenience offered by
cubic spline. Note that this class is implemented as a thin wrapper
over `scipy`'s :class:`~scipy.interpolate.CubicSpline`. Therefore, it
requires `scipy` to work
properly. :class:`~toppra.SplineInterpolator`'s usage is simple.


.. code-block:: python

   import toppra
   s_array = [0, 1, 2]
   wp_array = [(0, 0), (1, 2), (2, 0)]
   path = toppra.SplineInterpolator(s_array, wp_array)

That's it. To verify that is works correctly, let's try plotting. 

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   s_sampled = np.linspace(0, 2, 100)
   q_sampled = path.eval(s_sampled)
   plt.plot(s_sampled, q_sampled)
   plt.show()

The derivatives can be inspected by evaluating :func:`evald` and :func:`evaldd`
respectively.

Implement your own Interpolator
-------------------------------

TOPP-RA can handle custom Interpolators easily, as long as it conforms
to the abstract interface of :class:`toppra.Interpolator`. The most
important methods are obviously :func:`~toppra.Interpolator.eval`,
:func:`~toppra.Interpolator.evald` and :func:`~toppra.Interpolator.evaldd`.

All interpolators implemented in TOPP-RA are derived in this way. See
below for some examples.

Other Interpolators
--------------------

Other interpolators implemented in TOPP-RA:

1. :class:`toppra.RaveTrajectoryWrapper` A wrapper over `OpenRave`'s :class:`GenericTrajectory` class.
2. :class:`toppra.UnivariateSplineInterpolator` A wrapper over :class:`scipy.interpolate.UnivariateSpline`. This class implements smoothing spline. This means the resulting spline does not pass through all given waypoints. In contrast, `CubicSpline` implemented interpolating spline which passes through all given waypoints.
3. :class:`toppra.PolynomialInterpolator` A polynomial spline, implemented as a wrapper over :class:`numpy.polynomial.polynomial.Polynomial`.


