.. _module_ref:

Python API Reference
================================

.. automodule:: toppra.algorithm

Geometric paths
--------------------------------

.. automodule:: toppra.interpolator

.. autosummary::

   toppra.interpolator.AbstractGeometricPath
   toppra.SplineInterpolator
   toppra.RaveTrajectoryWrapper
   toppra.simplepath.SimplePath

AbstractGeometricPath
^^^^^^^^^^^^^^^^^^^

.. autoclass:: toppra.interpolator.AbstractGeometricPath
   :members: __call__, dof, path_interval, waypoints

SplineInterplator
^^^^^^^^^^^^^^^^^^^
.. autoclass:: toppra.SplineInterpolator
   :members: __call__, dof, path_interval, waypoints

RaveTrajectoryWrapper
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: toppra.RaveTrajectoryWrapper
   :members: __call__, dof, path_interval, waypoints

SimplePath
^^^^^^^^^^^^
.. autoclass:: toppra.simplepath.SimplePath
   :members: __call__, dof, path_interval

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
	       


