import numpy as np
import logging

logger = logging.getLogger(__name__)


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


def smooth_singularities(pp, us, xs, vs=None):
    """Smooth jitters due to singularities.

    Solving TOPP for discrete problem generated from collocation
    scheme tends to create jitters. This function finds and smooth
    them.

    Args:
    ----
    pp: PathParameterization
    us: ndarray
    xs: ndarray
    vs: ndarray, optional

    Returns:
    -------
    us_smth: ndarray,
    xs_smth: ndarray,
    vs_smth: ndarray,
    """
    # Find the indices
    singular_indices = []
    uds = np.diff(us, n=1)
    for i in range(pp.N - 3):
        if uds[i] < 0 and uds[i+1] > 0 and uds[i+2] < 0:
            logger.debug("Found potential singularity at {:d}".format(i))
            singular_indices.append(i)
    logger.debug("All singularities found: {}".format(singular_indices))

    # Smooth the singularities
    xs_smth = np.copy(xs)
    us_smth = np.copy(us)
    if vs is not None:
        vs_smth = np.copy(vs)
    for index in singular_indices:
        idstart = max(0, index)
        idend = min(pp.N, index + 4)
        xs_smth[range(idstart, idend + 1)] = (
            xs_smth[idstart] + (xs_smth[idend] - xs_smth[idstart]) *
            np.linspace(0, 1, idend + 1 - idstart))
        if vs is not None:
            data = [vs_smth[idstart] +
                    (xs_smth[idend] - xs_smth[idstart]) * frac
                    for frac in np.linspace(0, 1, idend + 1 - idstart)]
            vs_smth[range(idstart, idend + 1)] = np.array(data)

    for i in range(pp.N):
        us_smth[i] = (xs_smth[i+1] - xs_smth[i]) / 2 / (pp.ss[i+1] - pp.ss[i])

    if vs is not None:
        return us_smth, xs_smth, vs_smth
    else:
        return us_smth, xs_smth

