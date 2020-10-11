import numpy as np
from .linear_second_order import SecondOrderConstraint

class CartesianSpeedConstraint(SecondOrderConstraint):
    """This class implements a constraint on the linear & angular Cartesian speed of some part of the robot.
    """

    def __init__(self, fk, linear_speed_max, angular_speed_max, dof):
        """Initialize the constraint.

        Parameters
        ----------
        fk: (np.ndarray, np.ndarray) -> (linear_vel, angular_vel)
            The "FK" function that receives joint position and velocity as
            inputs and outputs the normal linear and angular speed of some part
            of the robot.
        linear_speed_max: float
            The max linear speed allowed for that link.
        angular_speed_max: float
            The max angular speed allowed for that link.
        dof: int
            The dimension of the joint position.
        """
        super(CartesianSpeedConstraint, self).__init__(self.invdyn, self.constraintf, self.constraintg, dof)
        self.fk = fk
        self.linear_speed_max = linear_speed_max
        self.angular_speed_max = angular_speed_max

    def invdyn(self, q, dq, ddq):
        linear_speed, angular_speed = self.fk(q, dq)
        return np.array([linear_speed**2, angular_speed**2])

    def constraintf(self, q):
        return np.identity(2)

    def constraintg(self, q):
        return np.array([self.linear_speed_max**2, self.angular_speed_max**2])
