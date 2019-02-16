A basic example
==================================================

The most basic usecase of TOPP-RA is compute time-optimal
time-parameterization subject to joint velocity and acceleration
constraints. This can be done very easily as shown below:

.. code-block:: python

   import toppra as ta
   import toppra.constraint as constraint
   import toppra.algorithm as algo
   import numpy as np
   import matplotlib.pyplot as plt
   import time

   # Random waypoints used to obtain a random geometric path. Here,
   # we use spline interpolation.
   dof = 6
   way_pts = np.random.randn(5, dof)
   path = ta.SplineInterpolator(np.linspace(0, 1, 5), way_pts)

   # Create velocity bounds, then velocity constraint object
   vlim_ = np.random.rand(dof) * 20
   vlim = np.vstack((-vlim_, vlim_)).T
   # Create acceleration bounds, then acceleration constraint object
   alim_ = np.random.rand(dof) * 2
   alim = np.vstack((-alim_, alim_)).T
   pc_vel = constraint.JointVelocityConstraint(vlim)
   pc_acc = constraint.JointAccelerationConstraint(
       alim, discretization_scheme=constraint.DiscretizationType.Interpolation)

   # Setup a parametrization instance
   instance = algo.TOPPRA([pc_vel, pc_acc], path, solver_wrapper='seidel')

   # Retime the trajectory, only this step is necessary.
   t0 = time.time()
   jnt_traj, aux_traj = instance.compute_trajectory(0, 0)
   print("Parameterization time: {:} secs".format(time.time() - t0))
   ts_sample = np.linspace(0, jnt_traj.get_duration(), 100)
   qdds_sample = jnt_traj.evaldd(ts_sample)

   plt.plot(ts_sample, qdds_sample)
   plt.xlabel("Time (s)")
   plt.ylabel("Joint acceleration (rad/s^2)")
   plt.show()


Solving any problem with TOPP-RA follows a simple process. There are
three steps. The first is initializing an
:class:`~toppra.Interpolator` to represents the geometric path of
interest. This is performed in the following lines.

.. code-block:: python

   way_pts = np.random.randn(N_samples, dof)
   path = ta.SplineInterpolator(np.linspace(0, 1, 5), way_pts)

Here, we simply choose random waypoints and define a geometric path as
the cubic spline interpolation over the waypoints using
:class:`~toppra.SplineInterpolator`.  Details on other kinds of
interpolator, as well as how to implement a new kind, will be
represented in :doc:`a later tutorial <1_geometric_path>`.

The second step is defining the constraints. In this example we
consider only joint velocity and joint acceleration constraints, both
of which require only the joint velocity and joint acceleration
limits.

.. code-block:: python

   # Create velocity bounds, then velocity constraint object
   vlim_ = np.random.rand(dof) * 20
   vlim = np.vstack((-vlim_, vlim_)).T
   # Create acceleration bounds, then acceleration constraint object
   alim_ = np.random.rand(dof) * 2
   alim = np.vstack((-alim_, alim_)).T
   pc_vel = constraint.JointVelocityConstraint(vlim)
   pc_acc = constraint.JointAccelerationConstraint(
       alim, discretization_scheme=constraint.DiscretizationType.Interpolation)

Note that a constraint can have `discretization_scheme` equals
`DiscretizationType.Interpolation` or
`DiscretizationType.Collocation`. Interpolation scheme is more
accurate but will be more computationally demanding. On the other
hand, Collocation scheme is faster. However, remark that for simple
problems the difference in computational time is negligible, while the
solution qualities differ greatly.

Other kinds of constraint can also be initialized and handled.
:doc:`This tutorial <2_can_linear_constraints>` discusses other kinds
of constraint, and show how to implement custom one.

Lastly, the third and final step is to solve the instance using
TOPP-RA.

.. code-block:: python

   instance = algo.TOPPRA([pc_vel, pc_acc], path, solver_wrapper='seidel')
   # Retime the trajectory, only this step is necessary.
   t0 = time.time()
   jnt_traj, aux_traj = instance.compute_trajectory(0, 0)
   print("Parameterization time: {:} secs".format(time.time() - t0))
   ts_sample = np.linspace(0, jnt_traj.get_duration(), 100)
   qdds_sample = jnt_traj.evaldd(ts_sample)


It is useful to remark that there are several solver wrappers
available. Each solver has a different run time characteristics, and
also, are applicable to different kinds or problems. For the simple
example considered here, `seidel` is the fastest. This solver also
comes bundled with TOPP-RA, without external dependencies.

Download the example given this tutorial here :download:`kinematics.py
<../../../examples/kinematics.py>`.
