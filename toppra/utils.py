import numpy as np


def compute_jacobian_wrench(robot, link, p):
    """ Compute the wrench Jacobian for link at point p.

    We look for J_wrench such that
          J_wrench.T * wrench = J_trans.T * F + J_rot.T * tau
    return the induced generalized joint torques.

    J_wrench is computed by stacking J_translation and J_rotation

    """
    J_trans = robot.ComputeJacobianTranslation(link.GetIndex(), p)
    J_rot = robot.ComputeJacobianAxisAngle(link.GetIndex())
    J_wrench = np.vstack((J_trans, J_rot))
    return J_wrench


def inv_dyn(rave_robot, q, qd, qdd, forceslist=None, returncomponents=True):
    """Simple wrapper around OpenRAVE's ComputeInverseDynamics function.

    M(q) qdd + C(q, qd) qd + g(q) = tau

    """
    if np.isscalar(q):  # Scalar case
        q_ = [q]
        qd_ = [qd]
        qdd_ = [qdd]
    else:
        q_ = q
        qd_ = qd
        qdd_ = qdd

    # Temporary remove velocity Limits
    vlim = rave_robot.GetDOFVelocityLimits()
    alim = rave_robot.GetDOFAccelerationLimits()
    rave_robot.SetDOFVelocityLimits(100 * vlim)
    rave_robot.SetDOFAccelerationLimits(100 * alim)
    with rave_robot:
        rave_robot.SetDOFValues(q_)
        rave_robot.SetDOFVelocities(qd_)
        res = rave_robot.ComputeInverseDynamics(
            qdd_, forceslist, returncomponents=returncomponents)

    rave_robot.SetDOFVelocityLimits(vlim)
    rave_robot.SetDOFAccelerationLimits(alim)
    return res
