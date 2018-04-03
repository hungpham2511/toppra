from enum import Enum
import logging
logger = logging.getLogger(__name__)

class ConstraintType(Enum):
    CanonicalLinear = 0

class Constraint(object):
    """ Base class for all parameterization constraints.

    A derived class should implement two methods:

    - get_constraint_type(): ConstraintType
    - compute_constraint_params(): tuple
    """

    def __repr__(self):
        string = self.__class__.__name__ + '(\n'
        string += '    Type: {:}'.format(self.get_constraint_type()) + '\n'
        string += self._format_string
        string += ')'
        return string

    def get_constraint_type(self):
        raise NotImplementedError

    def compute_constraint_params(self):
        raise NotImplementedError