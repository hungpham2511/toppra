.. _module_ref:

Module references
=========================

Interpolators
-------------


SplineInterplator
^^^^^^^^^^^^^^^^^^^
.. autoclass:: toppra.SplineInterpolator
   :members:

RaveTrajectoryWrapper
^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: toppra.RaveTrajectoryWrapper
   :members:

Interpolator (base class)
^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: toppra.Interpolator
   :members:

Constraints
------------

JointAccelerationConstraint
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: toppra.constraint.JointAccelerationConstraint
   :members:

JointVelocityConstraint
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: toppra.constraint.JointVelocityConstraint
   :members:

SecondOrderConstraint
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: toppra.constraint.CanonicalLinearSecondOrderConstraint
   :members:

CanonicalLinearConstraint (base class)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: toppra.constraint.CanonicalLinearConstraint
   :members:

RobustLinearConstraint
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: toppra.constraint.RobustCanonicalLinearConstraint
   :members:

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
	       
