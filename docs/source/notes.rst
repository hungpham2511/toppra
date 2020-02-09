.. _notes:


Notes on internals
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

Internals Overview
------------------------------------
