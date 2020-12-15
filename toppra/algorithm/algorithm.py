"""
toppra.algorithm.algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^

This module defines the abstract data types that define TOPP algorithms.

"""
from typing import Dict, Any, List, Tuple, Optional
import typing as T
import abc
import enum
import numpy as np
import time
import matplotlib.pyplot as plt

from toppra.constants import TINY
from toppra.interpolator import SplineInterpolator, AbstractGeometricPath
from toppra.constraint import Constraint
import toppra.interpolator as interpolator
import toppra.parametrizer as tparam

import logging

logger = logging.getLogger(__name__)


class ParameterizationData(object):
    """Internal data and output.
    """
    def __init__(self, *arg, **kwargs) -> None:
        self.return_code: ParameterizationReturnCode = ParameterizationReturnCode.ErrUnknown
        "ParameterizationReturnCode: Return code of the last parametrization attempt."
        self.gridpoints: Optional[np.ndarray] = None
        "np.ndarray: Shape (N+1, 1). Gridpoints"
        self.sd_vec: Optional[np.ndarray] = None
        "np.ndarray: Shape (N+1, 1). Path velocities"
        self.sdd_vec: Optional[np.ndarray] = None
        "np.ndarray: Shape (N+1, 1). Path acceleration"
        self.K: Optional[np.ndarray] = None
        "np.ndarray: Shape (N+1, 2). Controllable sets."
        self.X: Optional[np.ndarray] = None
        "np.ndarray: Shape (N+1, 2). Feasible sets."

    def __repr__(self):
        return "ParameterizationData(return_code:={}, N={:d})".format(
            self.return_code, self.gridpoints.shape[0])


class ParameterizationReturnCode(enum.Enum):
    """Return codes from a parametrization attempt.
    """
    Ok = "Ok: Successful parametrization"
    ErrUnknown = "Error: Unknown issue"
    ErrShortPath = "Error: Input path is very short"
    FailUncontrollable = "Error: Instance is not controllable"
    ErrForwardPassFail = "Error: Forward pass fail. Numerical errors occured"

    def __repr__(self):
        return super(ParameterizationReturnCode, self).__repr__()

    def __str__(self):
        return super(ParameterizationReturnCode, self).__repr__()


class ParameterizationAlgorithm(object):
    """Base parametrization algorithm class.

    This class specifies the generic behavior for parametrization algorithms.  For details on how
    to *construct* a :class:`ParameterizationAlgorithm` instance, as well as configure it, refer
    to the specific class.

    Example usage:

    .. code-block:: python

      # usage
      instance.compute_parametrization(0, 0)
      output = instance.problem_data

      # do this if you only want the final trajectory
      traj = instance.compute_trajectory(0, 0)

    .. seealso::

        :class:`toppra.algorithm.TOPPRA`,
        :class:`toppra.algorithm.TOPPRAsd`,
        :class:`~ParameterizationReturnCode`,
        :class:`~ParameterizationData`

    """

    def __init__(self, constraint_list, path, gridpoints=None, parametrizer=None,
                 gridpt_max_err_threshold: float=1e-3, gridpt_min_nb_points: int=100):
        self.constraints = constraint_list
        self.path = path  # Attr
        self._problem_data = ParameterizationData()
        # Handle gridpoints
        if gridpoints is None:
            gridpoints = interpolator.propose_gridpoints(
                path,
                max_err_threshold=gridpt_max_err_threshold,
                min_nb_points=gridpt_min_nb_points
            )
            logger.info(
                "No gridpoint specified. Automatically choose a gridpoint with %d points",
                len(gridpoints)
            )

        if (
            path.path_interval[0] != gridpoints[0]
            or path.path_interval[1] != gridpoints[-1]
        ):
            raise ValueError("Invalid manually supplied gridpoints.")
        self.gridpoints = np.array(gridpoints)
        self._problem_data.gridpoints = np.array(gridpoints)
        self._N = len(gridpoints) - 1  # Number of stages. Number of point is _N + 1
        for i in range(self._N):
            if gridpoints[i + 1] <= gridpoints[i]:
                logger.fatal("Input gridpoints are not monotonically increasing.")
                raise ValueError("Bad input gridpoints.")
        if parametrizer is None or parametrizer == "ParametrizeSpline":
            # TODO: What is the best way to type parametrizer?
            self.parametrizer: T.Any = tparam.ParametrizeSpline
        elif parametrizer == "ParametrizeConstAccel":
            self.parametrizer = tparam.ParametrizeConstAccel

    @property
    def constraints(self) -> List[Constraint]:
        """Constraints of interests."""
        return self._constraints

    @constraints.setter
    def constraints(self, value: List[Constraint]) -> None:
        # TODO: Validate constraints.
        self._constraints = value

    @property
    def problem_data(self) -> ParameterizationData:
        """Data obtained when solving the path parametrization."""
        return self._problem_data

    @abc.abstractmethod
    def compute_parameterization(self, sd_start: float, sd_end: float, return_data: bool=False):
        """Compute the path parameterization subject to starting and ending conditions.

        After this method terminates, the attribute
        :attr:`~problem_data` will contain algorithm output, as well
        as the result. This is the preferred way of retrieving problem
        output.

        Parameters
        ----------
        sd_start:
            Starting path velocity. Must be positive.
        sd_end:
            Goal path velocity. Must be positive.
        return_data:
            If true also return the problem data.

        """
        raise NotImplementedError

    def compute_trajectory(self, sd_start: float = 0, sd_end: float = 0) -> Optional[AbstractGeometricPath]:
        """Compute the resulting joint trajectory and auxilliary trajectory.

        This is a convenient method if only the final output is wanted.

        Parameters
        ----------
        sd_start:
            Starting path velocity.
        sd_end:
            Goal path velocity.
        return_data:
            If true, return a dict containing the internal data.

        Returns
        -------
        :
            Time-parameterized joint position trajectory or
            None If unable to parameterize. 

        """
        t0 = time.time()
        self.compute_parameterization(sd_start, sd_end)
        if self.problem_data.return_code != ParameterizationReturnCode.Ok:
            logger.warn("Fail to parametrize path. Return code: %s", self.problem_data.return_code)
            return None

        outputtraj = self.parametrizer(self.path, self.problem_data.gridpoints, self.problem_data.sd_vec)
        logger.info("Successfully parametrize path. Duration: %.3f, previously %.3f)",
                    outputtraj.path_interval[1], self.path.path_interval[1])
        logger.info("Finish parametrization in %.3f secs", time.time() - t0)
        return outputtraj

    def inspect(self, compute=True):
        """Inspect the problem internal data."""
        K = self.problem_data.K
        X = self.problem_data.X
        if X is not None:
            plt.plot(X[:, 0], c="green", label="Feasible sets")
            plt.plot(X[:, 1], c="green")
        if K is not None:
            plt.plot(K[:, 0], "--", c="red", label="Controllable sets")
            plt.plot(K[:, 1], "--", c="red")
        if self.problem_data.sd_vec is not None:
            plt.plot(self.problem_data.sd_vec ** 2, label="Velocity profile")
        plt.title("Path-position path-velocity plot")
        plt.xlabel("Path position")
        plt.ylabel("Path velocity square")
        plt.legend()
        plt.tight_layout()
        plt.show()


