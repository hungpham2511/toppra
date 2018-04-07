from enum import Enum
import logging
logger = logging.getLogger(__name__)


class ConstraintType(Enum):
    Unknown = -1
    CanonicalLinear = 0


class DiscretizationType(Enum):
    Collocation = 0
    Interpolation = 1


class Constraint(object):
    """ Base class for all parameterization constraints.

    This class has two main functions: first, to tell its type and second, to produce
    the parameters given the geometric path and the gridpoints.

    A derived class should implement the following method
    - compute_constraint_params(): tuple

    Attributes
    ----------
    constraint_type: ConstraintType
    discretization_type: DiscretizationType
    n_extra_vars: int

    """
    def __repr__(self):
        string = self.__class__.__name__ + '(\n'
        string += '    Type: {:}'.format(self.constraint_type) + '\n'
        string += '    Discretization Scheme: {:}'.format(self.discretization_type) + '\n'
        string += self._format_string
        string += ')'
        return string

    def get_no_extra_vars(self):
        return self.n_extra_vars

    def get_constraint_type(self):
        return self.constraint_type

    def get_discretization_type(self):
        raise self.discretization_type

    def compute_constraint_params(self, path, gridpoints):
        raise NotImplementedError
