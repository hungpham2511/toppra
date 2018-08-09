2: Canonical Linear Constraint
========================================

TOPP-RA *can not* handle arbitrary constraint; it can only process
specific kinds. This note concerns Second-Order constraint, a
constraint type that is supported and is relatively general. Many
commonly requested constraints, for example: joint acceleration and
Cartesian acceleration, can be reformulated to this form.

Background
------------------

So what do I mean by Second-Order constraint? A Second-Order
constraint is one that can be written in the following form

.. math::
   
   \mathbf A(q) \ddot q + q^\top \mathbf B(q) q + \mathbf c(q) = & \mathcal I (q, \dot q, \ddot q) ,\\
   \mathbf F(q) & \mathcal I (q, \dot q, \ddot q) \leq \mathbf g(q),

where :math:`\mathbf q` denotes the robot's joint position and
:math:`\mathbf{A, B, c, F, g}` are arbitrary, user-defined functions.

This constraint consists of two euations that are best viewed
independently.  The first equation is an *inverse dynamics* equation
that links the robot's joint quantities to the quantity of
interest. This could be joint torque, tool tip acceleration. Remark
that it can not be tool tip velocity. The second equation represents a
set of linear inequality on that quantity.

Code
--------------------------

TOPP-RA provides the class
:class:`~toppra.constraint.CanonicalLinearSecondOrderConstraint` that
accepts as its inputs:

1. the *inverse dynamics* function :math:`\mathcal I` and
2. the constraints functions :math:`\mathbf F, \mathbf q`

and returns an :class:`~toppra.constraint.CanonicalLinearConstraint`
object, readied to be used with TOPP-RA. See below for code example:

.. code-block:: python

   import toppra
   import numpy as np
   
   def inverse_dynamic(q, qd, qdd):
       # defined your function
       return value

   # Define a random path
   way_pts = np.random.randn(N_samples, dof)
   path = ta.SplineInterpolator(np.linspace(0, 1, 5), way_pts)

   c = toppra.constraint.SecondOrderConstraint(inverse_dynamic)
   instance = algo.TOPPRA([c], path)
   
   traj = instance.compute_trajectory()


Example: Cartesian Velocity and Acceleration constraints
---------------------------------------------------------


Example: Waiter Motion constraint
---------------------------------




