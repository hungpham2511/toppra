Modules reference
===============================


Interpolator
-------------

.. autoclass:: toppra.Interpolator
   :members:

.. autoclass:: toppra.SplineInterpolator
   :members:

.. autoclass:: toppra.RaveTrajectoryWrapper
   :members:


Constraints
------------

.. autoclass:: toppra.constraint.DiscretizationType
   :members:

Linear constraints
^^^^^^^^^^^^^^^^^^^

.. autoclass:: toppra.constraint.CanonicalLinearConstraint
   :members:

.. autoclass:: toppra.constraint.JointAccelerationConstraint
   :members:

.. autoclass:: toppra.constraint.JointVelocityConstraint
   :members:

Conic constraints
^^^^^^^^^^^^^^^^^^^

.. autoclass:: toppra.constraint.RobustCanonicalLinearConstraint
   :members:

Algorithms
------------

.. autoclass:: toppra.algorithm.TOPPRA
   :members: compute_feasible_sets, compute_controllable_sets, compute_parameterization



Solver Wrapper
----------------

.. autoclass:: toppra.solverwrapper.hot_qpoases_solverwrapper.hotqpOASESSolverWrapper
   :members:
	       
.. autoclass:: toppra.solverwrapper.cy_seidel_solverwrapper.seidelWrapper	       
   :members: solve_stagewise_optim

.. autoclass:: toppra.solverwrapper.ecos_solverwrapper.ecosWrapper
   :members: solve_stagewise_optim
	       
