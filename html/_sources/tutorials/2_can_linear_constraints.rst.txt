Simple [1]_ Second-Order Constraint
====================================

TOPP-RA *can not* account for arbitrary constraints; it can only
handle constraints of certain kinds. One class of constraints TOPP-RA
supports is Second-Order constraint.

This class of constraint is quite general.  Commonly found constraints
such as joint acceleration [2]_, tool tip Cartesian acceleration,
interaction force on the robot base, are all Second-Order constraints.

If you want to apply TOPP-RA to do motion planning on your robot,
which subject to one or multiple Second-Order constraints, read on.

Background
------------------

So what do I mean by Second-Order constraint? A Second-Order
constraint is one that can be written in the following form

.. math::
   
   \mathbf A(q) \ddot q + q^\top \mathbf B(q) q + \mathbf c(q) = & \mathcal I (q, \dot q, \ddot q) ,\\
   \mathbf F(q) & \mathcal I (q, \dot q, \ddot q) \leq \mathbf g(q),

where :math:`q` denotes the robot's joint position and
:math:`\mathbf{A, B, c, F, g}` are arbitrary functions. Remark that
you do not need to have an analytical expression for all these
functions. More will be said about this later.

This constraint consists of two equations. These equations are best
interpreted independently:

1. The first equation is an *inverse dynamics* equation that links the
   robot's joint quantities to the *quantity of interest* that is to
   be constrained. For example, joint torque or tool tip Cartesian
   acceleration.

2. The second equation represents a set of linear inequality on that
   quantity.

Defining a Second-Order constraint
--------------------------------------

To define a second-order constraint, you **need to be able to evaluate**
three functions:

1. Inverse-dynamic function :math:`\mathcal{I} (q, \dot q, \ddot q)`;

2. Two constraint functions :math:`\mathbf F(q)` and :math:`\mathbf g(q)`.

TOPP-RA receives these functions as inputs and produces a constraint
object that it can use to parametrize geometric paths.

Pseudo-code
--------------------------

To be concrete, TOPP-RA provides the class
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
       # define your function
       return I

   def F(q):
       # define your function
       return F_q

   def g(q):
       # define your function
       return g_q

   # Define a random path
   way_pts = np.random.randn(N_samples, dof)
   path = ta.SplineInterpolator(np.linspace(0, 1, 5), way_pts)

   c = toppra.constraint.CanonicalLinearSecondOrderConstraint(inverse_dynamic, F, g)
   instance = algo.TOPPRA([c], path)
   
   traj = instance.compute_trajectory()


Example: Cartesian Acceleration constraints
---------------------------------------------------------

*Note*: to run this example, OpenRAVE is required for kinematic
computations. However, it is easy to use any other programs.

Consider a 7-dof manipulator with joint velocity and acceleration
constraints. Suppose we want to constraint a point of interest (see
below figure) to have bounded Cartesian acceleration:

.. math:: 
   
   -0.5 \leq a_x, a_y, a_z \leq 0.5


.. image:: ../medias/2_figure_2b.png
   :align: center
   :height: 400px

Cartesian acceleration constraint is a Second-Order
constraint. Therefore, one can simply use the class
:class:`~toppra.constraint.CanonicalLinearSecondOrderConstraint` to
setup a suitable constraint object. The linear inequality is put into
the standard form as follows:

.. math:: 
   
   \begin{bmatrix}
   1 & 0&0 \\ 0& 1 &0 \\ 0&0& 1 \\
   -1 & 0&0 \\ 0& -1 &0 \\ 0&0& -1
   \end{bmatrix}
   \begin{bmatrix}
   a_x \\ a_y \\ a_z
   \end{bmatrix}
   \leq 
   \begin{bmatrix}
   0.5 \\ 
   0.5 \\ 
   0.5 \\ 
   0.5 \\ 
   0.5 \\ 
   0.5
   \end{bmatrix}
   
The left hand-side is :math:`\mathbf F(q)`. The right
hand-side is :math:`\mathbf g(q)`.
   
In the below code snippet, we use OpenRAVE to implement the inverse
kinematic function that returns :math:`[a_x, a_y, a_z]` given
:math:`q, \dot q, \ddot q`. Then, we initialize a constraint object
from these functions as shown in the below code. For your reference,
the final constraint object is :code:`pc_cart_acc`.

.. code-block:: python
   
    # setup Cartesian acceleration constraint to limit link 7
    # -0.5 <= a <= 0.5
    # Cartesian acceleration
    def inverse_dynamics(q, qd, qdd):
        with robot:
            vlim_ = robot.GetDOFVelocityLimits()
            robot.SetDOFVelocityLimits(vlim_ * 1000)  # remove velocity limits to compute stuffs
            robot.SetActiveDOFValues(q)
            robot.SetActiveDOFVelocities(qd)

            qdd_full = np.zeros(robot.GetDOF())
            qdd_full[:qdd.shape[0]] = qdd

            accel_links = robot.GetLinkAccelerations(qdd_full)
            robot.SetDOFVelocityLimits(vlim_)
        return accel_links[6][:3]  # only return the translational components

    F_q = np.zeros((6, 3))
    F_q[:3, :3] = np.eye(3)
    F_q[3:, :3] = -np.eye(3)
    g_q = np.ones(6) * 0.5
    def F(q):
        return F_q
    def g(q):
        return g_q

    pc_cart_acc = constraint.CanonicalLinearSecondOrderConstraint(
        inverse_dynamics, F, g, dof=7)


Using TOPP-RA to parametrize a given geometric path is
straightforward. There is no difference between this case and the
simple kinematic example, or any other situation.

.. code-block:: python

    all_constraints = [pc_vel, pc_acc, pc_cart_acc]
    instance = algo.TOPPRA(all_constraints, path, solver_wrapper='seidel')
    jnt_traj, _ = instance.compute_trajectory(0, 0)  # resulting trajectory

On my computer the whole process including evaluation of the dynamic
coefficients, which is quite costly, takes 5-8 ms.  Cartesian
acceleration of the resulting trajectory is plotted below.
	
.. image:: ../medias/2_figure_1.png
   
Download the example given this tutorial here
:download:`cartesian_accel.py <../../../examples/cartesian_accel.py>`.


.. [1] The form of Second-Order constraint presented in this tutorial
       is not the most general, hence, they are simple.

.. [2] Cartesian velocity constraint, as well as joint velocity
       constraint, are not Second-Order Constraint. They are
       First-Order constraints. These constraints will be treated in
       near future. For now, see how
       :class:`toppra.constraint.JointVelocityConstraint` is
       implemented.


