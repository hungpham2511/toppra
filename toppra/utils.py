"""
Some utility functions need to generate PathConstraints. Most are
specific to different scenarios.
"""
import logging
import functools
import warnings

import numpy as np


logger = logging.getLogger(__name__)


def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""

    # pylint: disable=C0111
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.warn(
            "Call to deprecated function {} in module {}.".format(
                func.__name__, func.__module__
            ),
            category=DeprecationWarning,
        )
        return func(*args, **kwargs)

    return new_func


def setup_logging(level="WARN"):
    """Setup basic logging facility to console.
    """
    logger_toppra = logging.getLogger("toppra")
    handler_basic = logging.StreamHandler()
    handler_basic.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(levelname)5s [%(filename)s : %(lineno)d] %(message)s")
    handler_basic.setFormatter(formatter)
    logger_toppra.setLevel(level)
    logger_toppra.addHandler(handler_basic)


def compute_jacobian_wrench(robot, link, point):
    """Compute the wrench Jacobian for `link` at `point`.

    We look for J_wrench such that
          J_wrench.T * wrench = J_trans.T * F + J_rot.T * tau
    return the induced generalized joint torques.

    J_wrench is computed by stacking J_translation and J_rotation

    """
    jacobian_translation = robot.ComputeJacobianTranslation(link.GetIndex(), point)
    jacobian_rotation = robot.ComputeJacobianAxisAngle(link.GetIndex())
    jacobian_wrench = np.vstack((jacobian_translation, jacobian_rotation))
    return jacobian_wrench


def inv_dyn(rave_robot, q, qd, qdd, forceslist=None, returncomponents=True):
    """Inverse dynamics equation.

    Simple wrapper around OpenRAVE's ComputeInverseDynamics
    function. Return the numerical values of the components of the
    inverse dynamics equation.

          M(q) qdd + C(q, qd) qd + g(q)
        = t1 + t2 + t3

    Parameters
    ----------
    rave_robot : OpenRAVE.robot
    q : (_N, ) ndarray
        Joint position.
    qd : (_N, ) ndarray
        Joint velocity.
    qdd : (_N, ) ndarray
        Joint acceleration.
    returncomponents : Bool
        If True, return the list [t1, t2, t3]
        If False, return t1 + t2 + t3

    Returns
    -------
    res : (3, ) List, or ndarray
        See returncomponents parameter.
    """
    if np.isscalar(q):  # Scalar case
        _q = [q]
        _qd = [qd]
        _qdd = [qdd]
    else:
        _q = q
        _qd = qd
        _qdd = qdd

    # Temporary remove kinematic Limits
    vlim = rave_robot.GetDOFVelocityLimits()
    alim = rave_robot.GetDOFAccelerationLimits()
    rave_robot.SetDOFVelocityLimits(100 * vlim)
    rave_robot.SetDOFAccelerationLimits(100 * alim)
    # Do computation
    with rave_robot:
        rave_robot.SetDOFValues(_q)
        rave_robot.SetDOFVelocities(_qd)
        res = rave_robot.ComputeInverseDynamics(
            _qdd, forceslist, returncomponents=returncomponents
        )
    # Restore kinematic limits
    rave_robot.SetDOFVelocityLimits(vlim)
    rave_robot.SetDOFAccelerationLimits(alim)
    return res


def smooth_singularities(parametrization_instance, us, xs, vs=None):
    """Smooth jitters due to singularities.

    Solving TOPP for discrete problem generated from collocation
    scheme tends to create jitters. This function finds and smooths
    them.

    Notes
    -----
    (`us_smth`, `xs_smth`) is a *valid* path-parameterization. They
    satisfy the linear continuity condition :math:`x_{i+1} = x_i + 2 delta_i u_i`.

    This function is safe: it will always return a solution.

    Parameters
    ----------
    parametrization_instance: :class:`.qpOASESPPSolver`
    us: array
        Shape (_N, ). Controls.
    xs: array
        Shape (_N+1, ). Squared velocities.
    vs: array, optional
        If not given, `vs_smth` will not be returned.

    Returns
    -------
    us_smth: array
        Shape (_N, ). Smoothed controls.
    xs_smth: array
        Shape (_N+1, ). Smoothed squared velocities.
    vs_smth: array
        If `vs` is not given, `vs_smth` will not be returned.

    """
    # Find the indices
    singular_indices = []
    uds = np.diff(us, n=1)
    for i in range(parametrization_instance.N - 3):
        if uds[i] < 0 < uds[i + 1] and uds[i + 2] < 0:
            logger.debug("Found potential singularity at {:d}".format(i))
            singular_indices.append(i)
    logger.debug("Found singularities at %s", singular_indices)

    # Smooth the singularities
    xs_smth = np.copy(xs)
    us_smth = np.copy(us)
    if vs is not None:
        vs_smth = np.copy(vs)
    for index in singular_indices:
        idstart = max(0, index)
        idend = min(parametrization_instance.N, index + 4)
        xs_smth[range(idstart, idend + 1)] = xs_smth[idstart] + (
            xs_smth[idend] - xs_smth[idstart]
        ) * np.linspace(0, 1, idend + 1 - idstart)
        if vs is not None:
            data = [
                vs_smth[idstart] + (xs_smth[idend] - xs_smth[idstart]) * frac
                for frac in np.linspace(0, 1, idend + 1 - idstart)
            ]
            vs_smth[range(idstart, idend + 1)] = np.array(data)

    for i in range(parametrization_instance.N):
        us_smth[i] = (
            (xs_smth[i + 1] - xs_smth[i])
            / 2
            / (parametrization_instance.ss[i + 1] - parametrization_instance.ss[i])
        )

    if vs is not None:
        return us_smth, xs_smth, vs_smth
    return us_smth, xs_smth
