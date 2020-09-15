"""
TOPP-RA
=======
Reachability-Analysis-based Time-Optimal Path Parametrization.

This package produces routines for creation and handling path constraints
using the algorithm `TOPP-RA`.
"""
import logging

# core modules
from .interpolator import RaveTrajectoryWrapper, SplineInterpolator,\
    UnivariateSplineInterpolator, PolynomialPath
from .simplepath import SimplePath
from .parametrizer import ParametrizeConstAccel, ParametrizeSpline
from . import constraint
from . import algorithm
from . import solverwrapper

# utility
from .utils import smooth_singularities, setup_logging
from .planning_utils import retime_active_joints_kinematics,\
    create_rave_torque_path_constraint

# set nullhandler by default
logging.getLogger('toppra').addHandler(logging.NullHandler())
