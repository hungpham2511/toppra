"""Base class for all path parametrization contraints. """
from enum import Enum
import logging
logger = logging.getLogger(__name__)


class ConstraintType(Enum):
    """Type of path parametrization constraint."""
    Unknown = -1
    CanonicalLinear = 0
    CanonicalConic = 1


class DiscretizationType(Enum):
    """Enum to mark different Discretization Scheme for constraint.

    1. `Collocation`: smaller problem size, but lower accuracy.
    2. `Interplation`: larger problem size, but higher accuracy.

    In general, the difference in speed is not too large. Should use
    Interpolation if possible.
    """
    Collocation = 0
    Interpolation = 1


class Constraint(object):
    """The base constraint class."""
    def __repr__(self):
        string = self.__class__.__name__ + '(\n'
        string += '    Type: {:}'.format(self.constraint_type) + '\n'
        string += '    Discretization Scheme: {:}'.format(self.discretization_type) + '\n'
        string += self._format_string
        string += ')'
        return string

    def get_dof(self):
        return self.dof

    def get_no_extra_vars(self):
        return self.n_extra_vars

    def get_constraint_type(self):
        return self.constraint_type

    def get_discretization_type(self):
        return self.discretization_type

    def set_discretization_type(self, t):
        """Discretization type: Collocation or Interpolation.

        Parameters
        ----------
        t: int, or DiscretizationType
            If is 1, set to Interpolation.
            If is 0, set to Collocation.
        """
        if t == 0:
            self.discretization_type = DiscretizationType.Collocation
        elif t == 1:
            self.discretization_type = DiscretizationType.Interpolation
        elif t == DiscretizationType.Collocation or t == DiscretizationType.Interpolation:
            self.discretization_type = t
        else:
            raise "Discretization type: {:} not implemented!".format(t)

    def compute_constraint_params(self, path, gridpoints, scaling):
        raise NotImplementedError
