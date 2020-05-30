"""Abstract types for parametrization algorithm.
"""
from typing import Dict, Any, List, Tuple
import abc
import enum
import numpy as np

from ..constants import TINY
from toppra.interpolator import SplineInterpolator, AbstractGeometricPath
from toppra.constraint import Constraint
import toppra.interpolator as interpolator
import toppra.parametrize_const_accel as tparam

import logging

logger = logging.getLogger(__name__)


class ParameterizationData(object):
    """Internal data and output.
    """
    def __init__(self, *arg, **kwargs) -> None:
        self.return_code: ParameterizationReturnCode = ParameterizationReturnCode.ErrUnknown
        "ParameterizationReturnCode: Return code of the last parametrization attempt."
        self.gridpoints: np.ndarray = None
        "np.ndarray: Shape (N+1, 1). Gridpoints"
        self.sd_vec: np.ndarray = None
        "np.ndarray: Shape (N+1, 1). Path velocities"
        self.sdd_vec: np.ndarray = None
        "np.ndarray: Shape (N+1, 1). Path acceleration"
        self.K: np.ndarray = None
        "np.ndarray: Shape (N+1, 2). Controllable sets."
        self.X: np.ndarray = None
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

    def __init__(self, constraint_list, path, gridpoints=None):
        self.constraints = constraint_list
        self.path = path  # Attr
        self._problem_data = ParameterizationData()
        # Handle gridpoints
        if gridpoints is None:
            gridpoints = interpolator.propose_gridpoints(path, max_err_threshold=1e-3)
            logger.info(
                "No gridpoint specified. Automatically choose a gridpoint. See `propose_gridpoints`."
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
        """Intermediate data obtained while solving the path parametrization.. """
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

    def compute_trajectory(self, sd_start: float = 0, sd_end: float = 0, return_data: bool =
                           False) -> Tuple[AbstractGeometricPath, AbstractGeometricPath]:
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
            A 2-tuple. The first element is the time-parameterized joint position trajectory or
            None If unable to parameterize. The second element is the
            time-parameterized auxiliary variable trajectory. Is None if
            unable to parameterize

        """
        sdd_grid, sd_grid, v_grid, K = self.compute_parameterization(
            sd_start, sd_end, return_data=True
        )
        if self.problem_data.return_code != ParameterizationReturnCode.Ok:
            logger.warn("Fail to parametrize path. Return code: %s", self.problem_data.return_code)
            return None

        return tparam.ParametrizeSpline(self.path, self.problem_data.gridpoints, self.problem_data.sd_vec)
