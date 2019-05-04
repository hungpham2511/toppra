"""
TOPP-RA
=======
Reachability-Analysis-based Time-Optimal Path Parametrization.

This package produces routines for creation and handling path constraints
using the algorithm `TOPP-RA`.
"""

from .interpolator import RaveTrajectoryWrapper, SplineInterpolator,\
    UnivariateSplineInterpolator, PolynomialPath, Interpolator
from .utils import smooth_singularities, setup_logging
from .planning_utils import retime_active_joints_kinematics,\
    create_rave_torque_path_constraint

from . import constraint
from . import algorithm
from . import solverwrapper

# set nullhandler by default
import logging
logging.getLogger('toppra').addHandler(logging.NullHandler())
