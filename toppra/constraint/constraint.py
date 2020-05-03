"""Base class for all path parametrization contraints. """
from enum import Enum
import logging
import numpy as np
from ..interpolator import AbstractGeometricPath

logger = logging.getLogger(__name__)


class ConstraintType(Enum):
    """Type of path parametrization constraint."""

    #: Unknown
    Unknown = -1
    #: Simple linear constraints with only linear inequalities
    CanonicalLinear = 0
    #: Linear constraints with linear conic inequalities.
    CanonicalConic = 1


class DiscretizationType(Enum):
    """Enum to mark different Discretization Scheme for constraint.

    In general, the difference in speed is not too large. Should use
    Interpolation if possible.
    """
    #: Smaller problem size, but lower accuracy.
    Collocation = 0

    #: Larger problem size, but higher accuracy.
    Interpolation = 1


class Constraint(object):
    """The base constraint class."""

    def __repr__(self):
        string = self.__class__.__name__ + "(\n"
        string += "    Type: {:}".format(self.constraint_type) + "\n"
        string += (
            "    Discretization Scheme: {:}".format(self.discretization_type) + "\n"
        )
        string += self._format_string
        string += ")"
        return string

    def get_dof(self):
        """Return the degree of freedom of the constraint.

        TODO: It is unclear what is a dof of a constraint. Perharps remove this.
        """
        return self.dof

    def get_no_extra_vars(self):
        """Return the number of extra variable required.

        TODO: This is not a property of a constraint. Rather it is specific to kinds of constraints. To be removed.
        """
        return self.n_extra_vars

    def get_constraint_type(self):
        """Return the constraint type.

        TODO: Use property instead.

        """
        return self.constraint_type

    def get_discretization_type(self):
        """Return the discretization type.

        TODO: Use property instead.

        """
        return self.discretization_type

    def set_discretization_type(self, discretization_type):
        """Discretization type: Collocation or Interpolation.

        Parameters
        ----------
        discretization_type: int or :class:`DiscretizationType`
            Method to discretize this constraint.
        """
        if discretization_type == 0:
            self.discretization_type = DiscretizationType.Collocation
        elif discretization_type == 1:
            self.discretization_type = DiscretizationType.Interpolation
        elif (
            discretization_type == DiscretizationType.Collocation
            or discretization_type == DiscretizationType.Interpolation
        ):
            self.discretization_type = discretization_type
        else:
            raise "Discretization type: {:} not implemented!".format(
                discretization_type
            )

    def compute_constraint_params(
        self, path: AbstractGeometricPath, gridpoints: np.ndarray, *args, **kwargs
    ):
        """Evaluate parameters of the constraint."""
        raise NotImplementedError
