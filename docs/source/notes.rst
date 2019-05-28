.. _notes:

Notes
========

.. _derivationKinematics:

Derivation of kinematical quantities
------------------------------------

In `toppra` we deal with geometric paths, which are often represented
as a function :math:`\mathbf p(s)`. Here :math:`s` is the path
position and usually belongs to the interval :math:`[0, 1]`. Notice
that `toppra` can also handle arbitrary interval.


Important expression relating kinematic quantities:

.. math::
   \mathbf q(t) &= \mathbf p(s(t)) \\
   \dot{\mathbf p}(t) &= \mathbf p'(s) \dot s(t) \\
   \ddot{\mathbf p}(t) &= \mathbf p'(s) \ddot s(t) + \mathbf p''(s) \dot s(t)^2

