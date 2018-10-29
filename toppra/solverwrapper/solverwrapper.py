import numpy as np


class SolverWrapper(object):
    """The base class for all solver wrappers.

    Solver wrappers implement a core method needed by all
    Reachability-based algorithms: The method
    `solve_stagewise_optim`. This methods solves a Linear/Quadratic
    Program subject to linear constraints at a given stage, and
    possibly with additional auxiliary constraints.

    Note that some solver wrappers only handle Linear Program while
    some handle both.

    Certain solver wrappers need to be setup and close down before and
    after usage. For instance, the wrappers for mosek and qpOASES with
    warmstart capability. To provide this functionality, this class
    contains two abstract methods `setup_solver` and `close_solver`,
    which should be called before and after any call to
    `solve_stagewise_optim`, so that necessary setups can be made.

    Attributes
    ----------
    constraints : list of `Constraint`
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
        # path scaling: intuitively, if this value is not 1, the TOPP
        # problem will be solved as if the input path is scaled linearly.
        self.scaling = self.path_discretization[-1] / self.path.get_duration()
        # End main attributes
        self.N = len(path_discretization) - 1  # Number of stages. Number of point is _N + 1
        self.deltas = self.path_discretization[1:] - self.path_discretization[:-1]
        for i in range(self.N):
            assert path_discretization[i + 1] > path_discretization[i]

        self.params = [
            c.compute_constraint_params(self.path, self.path_discretization, self.scaling)
            for c in self.constraints]
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
        H: (d,d)array or None
            The coefficient of the quadratic objective function. If is
            None, neglect the quadratic term.
        g: (d,)array
            The linear term.
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
        double array, or list
             If successes, return an array containing the optimal
             variable.  Since NaN is also a valid double, this list
             contains NaN if the optimization problem is infeasible.
        """
        raise NotImplementedError

    def setup_solver(self):
        pass

    def close_solver(self):
        pass
