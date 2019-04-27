.. _module_ref:

Module references
=========================

Interpolators
-------------

Interpolator base class
^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: toppra.Interpolator
   :members:

Spline Interplator
^^^^^^^^^^^^^^^^^^^
.. autoclass:: toppra.SplineInterpolator
   :members:

Rave Trajectory Wrapper
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: toppra.RaveTrajectoryWrapper
   :members:

Constraints
------------

Joint Acceleration Constraint
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: toppra.constraint.JointAccelerationConstraint
   :members:
   :special-members:

Joint Velocity Constraint
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: toppra.constraint.JointVelocityConstraint
   :members:
   :special-members:

Second Order Constraint
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: toppra.constraint.SecondOrderConstraint
   :members:
   :special-members:

Robust Linear Constraint
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: toppra.constraint.RobustLinearConstraint
   :members:

Canonical Linear Constraint (base class)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: toppra.constraint.LinearConstraint
   :members:
   :special-members:


Constraints (base class)
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: toppra.constraint.Constraint

DiscretizationType (enum)
^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: toppra.constraint.DiscretizationType
   :members:

Algorithms
------------

TOPPRA (time-optimal)
^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: toppra.algorithm.TOPPRA
   :members: compute_parameterization, compute_trajectory, compute_feasible_sets, compute_controllable_sets

TOPPRAsd (specific-duration)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: toppra.algorithm.TOPPRAsd
   :members: set_desired_duration, compute_parameterization, compute_trajectory, compute_feasible_sets, compute_controllable_sets

Solver Wrapper
----------------

All computations in TOPP-RA algorithms are done by the linear and
quadratic solvers, wrapped in solver wrappers.

qpOASES (with hot-start)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: toppra.solverwrapper.hotqpOASESSolverWrapper
   :members: close_solver, setup_solver, solve_stagewise_optim

seidel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: toppra.solverwrapper.seidelWrapper	       
   :members: solve_stagewise_optim

ecos
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: toppra.solverwrapper.ecosWrapper
   :members: solve_stagewise_optim
	       
