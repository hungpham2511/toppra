"""Modules implementing different dynamics constraints.

Base abstractions and enum
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: toppra.constraint.Constraint
   :members: compute_constraint_params, set_discretization_type, __repr__

.. autoclass:: toppra.constraint.ConstraintType
   :members:

.. autoclass:: toppra.constraint.LinearConstraint
   :members:
   :special-members:
   :show-inheritance:

.. autoclass:: toppra.constraint.DiscretizationType
   :members:

Velocity Constraints
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: toppra.constraint.JointVelocityConstraint
   :members:
   :special-members:
   :show-inheritance:

.. autoclass:: toppra.constraint.JointVelocityConstraintVarying
   :members:
   :special-members:
   :show-inheritance:

Second Order Constraints
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: toppra.constraint.SecondOrderConstraint
   :members:
   :special-members:
   :show-inheritance:

.. autoclass:: toppra.constraint.JointTorqueConstraint
   :members:
   :special-members:
   :show-inheritance:

.. autoclass:: toppra.constraint.JointAccelerationConstraint
   :members:
   :special-members:
   :show-inheritance:

Robust Linear Constraint
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: toppra.constraint.RobustLinearConstraint
   :members:
   :show-inheritance:

Misc
^^^^^^^^^^
.. autofunction:: toppra.constraint.canlinear_colloc_to_interpolate

"""
from .constraint import ConstraintType, DiscretizationType, Constraint
from .joint_torque import JointTorqueConstraint
from .linear_joint_acceleration import JointAccelerationConstraint
from .linear_joint_velocity import (
    JointVelocityConstraint,
    JointVelocityConstraintVarying,
)
from .linear_second_order import SecondOrderConstraint, canlinear_colloc_to_interpolate
from .conic_constraint import RobustLinearConstraint
from .linear_constraint import LinearConstraint
