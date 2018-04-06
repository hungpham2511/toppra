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

    A derived class should implement the following methods:

    - get_constraint_type(): ConstraintType
    - get_discretization_type(): DiscretizationType
    - compute_constraint_params(): tuple

    """
    def __repr__(self):
        string = self.__class__.__name__ + '(\n'
        string += '    Type: {:}'.format(self.constraint_type) + '\n'
        string += '    Discretization Scheme: {:}'.format(self.discretization_type) + '\n'
        string += self._format_string
        string += ')'
        return string

    def get_constraint_type(self):
        return self.constraint_type

    def get_discretization_type(self):
        return self.discretization_type

    def compute_constraint_params(self, path, gridpoints):
        """ Compute the parameters of the constraint.

        """
        raise NotImplementedError

    def get_no_extra_vars(self):
        return self.n_extra_vars