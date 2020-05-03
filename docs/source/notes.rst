.. _notes:


Advanced usage
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

.. _module_ref:

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
	       
