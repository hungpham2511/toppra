"""
Constraints
----------------------

Modules implementing different dynamics constraints.

ConstraintType
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: toppra.constraint.ConstraintType
   :members:

DiscretizationType
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: toppra.constraint.DiscretizationType
   :members:

LinearConstraint
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: toppra.constraint.LinearConstraint
   :members:
   :special-members:
   :show-inheritance:


JointVelocityConstraint
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: toppra.constraint.JointVelocityConstraint
   :members:
   :special-members:
   :show-inheritance:

JointVelocityConstraintVarying
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: toppra.constraint.JointVelocityConstraintVarying
   :members:
   :special-members:
   :show-inheritance:

SecondOrderConstraints
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: toppra.constraint.SecondOrderConstraint
   :members:
   :special-members:
   :show-inheritance:

JointTorqueConstraint
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: toppra.constraint.JointTorqueConstraint
   :members:
   :special-members:
   :show-inheritance:

JointAccelerationConstraint
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: toppra.constraint.JointAccelerationConstraint
   :members:
   :special-members:
   :show-inheritance:

RobustLinearConstraint
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: toppra.constraint.RobustLinearConstraint
   :members:
   :show-inheritance:

[abstract]Constraint
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: toppra.constraint.Constraint
   :members: compute_constraint_params, set_discretization_type, __repr__

[internal]Methods
^^^^^^^^^^^^^^^^^^^^^^^^
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
