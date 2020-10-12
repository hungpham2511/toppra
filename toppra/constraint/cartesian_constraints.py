import numpy as np
from .linear_second_order import SecondOrderConstraint

class CartesianSpeedConstraint(SecondOrderConstraint):
    """
    This class implements a constraint on the magnitudes of the linear & angular
    Cartesian velocity vectors of one of the robot's parts (link, joint, etc).
    
    The forward kinematic velocity is be provided via a callback, which makes
    this constraint agnostic of the robot's geometry and agnostic of whatever
    FK algorithm is used.
    """

    def __init__(self, fk, linear_speed_max, angular_speed_max, dof):
        """Initialize the constraint.

        Parameters
        ----------
        fk: (np.ndarray, np.ndarray) -> (float, float)
            The "FK" function that receives joint positions and velocities as
            inputs and outputs the magnitude of the linear and angular
            velocity vectors for some the monitored part of the robot.
        linear_speed_max: float
            The max linear speed allowed for the monitored part.
        angular_speed_max: float
            The max angular speed allowed for the monitored part.
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
