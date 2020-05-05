.. _module_ref:

Python API Reference
================================

Path-parametrization Algorithms
--------------------------------

.. automodule:: toppra.algorithm

TOPPRA (time-optimal)
^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: toppra.algorithm.TOPPRA
   :members: problem_data, compute_parameterization, compute_trajectory, compute_feasible_sets, compute_controllable_sets

TOPPRAsd (specific-duration)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: toppra.algorithm.TOPPRAsd
   :members: problem_data, set_desired_duration, compute_parameterization, compute_trajectory, compute_feasible_sets, compute_controllable_sets


Geometric paths
--------------------------------

.. automodule:: toppra.interpolator

.. autoclass:: toppra.interpolator.AbstractGeometricPath
   :members: __call__, dof, path_interval, waypoints


Spline Interplator
^^^^^^^^^^^^^^^^^^^
.. autoclass:: toppra.SplineInterpolator
   :members: __call__, dof, path_interval, waypoints

Rave Trajectory Wrapper
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: toppra.RaveTrajectoryWrapper
   :members: __call__, dof, path_interval, waypoints

.. autofunction:: toppra.interpolator.propose_gridpoints

Constraints
--------------

.. automodule:: toppra.constraint

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
	       


C++ API Reference
================================

TBD
