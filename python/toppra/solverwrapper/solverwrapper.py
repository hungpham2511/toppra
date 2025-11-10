"""This module provides different solverwrapper implementations."""
import logging
import numpy as np

logger = logging.getLogger(__name__)


# pylint: disable=unused-import
def available_solvers(output_msg=True):
    """Check for available solvers."""
    try:
        import ecos

        IMPORT_ECOS = True
    except ImportError as err:
        IMPORT_ECOS = False
    try:
        import qpoases

        IMPORT_QPOASES = True
    except ImportError as err:
        IMPORT_QPOASES = False
    try:
        import cvxpy

        IMPORT_CVXPY = True
    except ImportError as err:
        IMPORT_CVXPY = False
    solver_availability = (
        ("seidel", True),
        ("hotqpoases", IMPORT_QPOASES),
        ("qpoases", IMPORT_QPOASES),
        ("ecos", IMPORT_ECOS),
        ("cvxpy", IMPORT_CVXPY),
    )

    if output_msg:
        print(solver_availability)
    return solver_availability

def check_solver_availability(solver):
    check = available_solvers(False)
    for sname, avail in check:
        if sname == solver and avail:
            return True
    return False


class SolverWrapper(object):
    """The base class for all solver wrappers.

    All SolverWrapper have to implement a core method needed by all
    Reachability Analysis-based algorithms:
    `solve_stagewise_optim`. This methods solves a Linear/Quadratic
    Program subject to linear constraints at the given stage, and
    possibly with additional auxiliary constraints.

    Note that some SolverWrappers only handle Linear Program while
    some handle both.

    Certain solver wrappers need to be setup and close down before and
    after usage. For instance, the wrappers for mosek and qpOASES with
    warmstart capability. To provide this functionality, this class
    contains two abstract methods `setup_solver` and `close_solver`,
    which should be called before and after any call to
    `solve_stagewise_optim`, so that necessary setups can be made.

    Each solver wrapper should provide solver-specific constraint,
    such as ultimate bound the variable u, x. For some solvers such as
    ECOS, this is very important.

    Attributes
    ----------
    constraints : :class:`toppra.Constraint` []
        Constraints on the robot system.
    path : Interpolator
        The geometric path to be time-parametrized.
    path_discretization: array
        The discretization grid use to discretize the geometric path.

    """

    def __init__(self, constraint_list, path, path_discretization):
        # Main attributes
        self.constraints = constraint_list
        self.path = path
        self.path_discretization = np.array(path_discretization)
        # End main attributes
        self.N = (
            len(path_discretization) - 1
        )  # Number of stages. Number of point is _N + 1
        self.deltas = self.path_discretization[1:] - self.path_discretization[:-1]
        for i in range(self.N):
            assert path_discretization[i + 1] > path_discretization[i]

        self.params = [
            c.compute_constraint_params(
                self.path, self.path_discretization
            )
            for c in self.constraints
        ]
        self.nV = 2 + sum([c.get_no_extra_vars() for c in self.constraints])

    def get_no_stages(self):
        """Return the number of stages.

        The number of gridpoints equals N + 1, where N is the number
        of stages.
        """
        return self.N

    def get_no_vars(self):
        """ Return total number of variables, including u, x.
        """
        return self.nV

    def get_deltas(self):
        return self.deltas

    def solve_stagewise_optim(self, i, H, g, x_min, x_max, x_next_min, x_next_max):
        """Solve a stage-wise quadratic (or linear) optimization problem.

        The quadratic optimization problem is described below:

        .. math::
            \\text{min  }  & 0.5 [u, x, v] H [u, x, v]^\\top + [u, x, v] g    \\\\
            \\text{s.t.  } & [u, x] \\text{ is feasible at stage } i \\\\
                           & x_{min} \leq x \leq x_{max}             \\\\
                           & x_{next, min} \leq x + 2 \Delta_i u \leq x_{next, max},

        where `v` is an auxiliary variable, only exist if there are
        non-canonical constraints.  The linear program is the
        quadratic problem without the quadratic term.

        Parameters
        ----------
        i: int
            The stage index.
        H: :class:`~numpy:numpy.ndarray` or None
            Shape (d, d). The coefficient of the quadratic objective
            function. If is None, set the quadratic term to zero.
        g: :class:`~numpy:numpy.ndarray`
            Shape (d,). The linear term.
        x_min: float
            If not specified, set to NaN.
        x_max: float
            If not specified, set to NaN.
        x_next_min: float
            If not specified, set to NaN.
        x_next_max: float
            If not specified, set to NaN.

        Returns
        -------
        double :class:`~numpy:numpy.ndarray`
             If successes, return a double array containing the optimal
             variable.  Otherwise return a array that contains `numpy.nan`.

        """
        raise NotImplementedError

    def setup_solver(self):
        pass

    def close_solver(self):
        pass
