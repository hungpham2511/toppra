import numpy as np

class SolverWrapper(object):
    """ Base class for all solver wrappers.

    Solver wrappers are used only for implementing Reachability-based parameterization algorithms.
    The main public interface of this class is the method `solve_stagewise_optim`, which needs to
    be implemented by derived classes.
    """

    def __init__(self, constraint_list, path, path_discretization):
        self.N = len(path_discretization) - 1  # Number of stages. Number of point is N + 1
        self.path = path
        self.constraints = constraint_list
        self.path_discretization = np.array(path_discretization)
        assert path.get_path_interval()[0] == path_discretization[0]
        assert path.get_path_interval()[1] == path_discretization[-1]
        for i in range(self.N):
            assert path_discretization[i + 1] > path_discretization[i]

        self.params = [c.compute_constraint_params(self.path, self.path_discretization)
                       for c in self.constraints]

    def solve_stagewise_optim(self, i, H, g, x_min, x_max, x_next_min, x_next_max):
        """ Solve a stage-wise quadratic optimization.

        Parameters
        ----------
        i: int
            For the meaning of the parameters, see notes.
        H: array
        g: array
        x_min: float
        x_max: float
        x_next_min: float
        x_next_max: float

        Returns
        -------
        array or None
             If the optimization successes, return an array containing the optimal variable.
             Otherwise, return None.

        Notes
        -----
        This is the main public interface of `SolverWrapper`. The stage-wise quadratic optimization problem
        is given by:

        .. math::
            \\text{min  }  & 0.5 [u, x, v] H [u, x, v]^\\top + [u, x, v] g    \\\\
            \\text{s.t.  } & [u, x] \\text{ is feasible at stage } i \\\\
                           & x_{min} \leq x \leq x_{max}             \\\\
                           & x_{next, min} \leq x + 2 \Delta_i u \leq x_{next, max},

        where `v` is an auxiliary variable, only presented in non-canonical constraints.
        """
        raise NotImplementedError