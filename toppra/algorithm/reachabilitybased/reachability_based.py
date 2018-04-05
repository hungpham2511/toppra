from ..algorithm import ParameterizationAlgorithm
from ...solverwrapper import cvxpyWrapper
import numpy as np

class ReachabilityAlgorithm(ParameterizationAlgorithm):
    """ Base class for all Reachability Analysis-based parameterization algorithms.

    In contrast to a generic path parameterization algorithm, a RA-based algorithm
    implement additionally three methods:
    - compute_controllable_sets
    - compute_reachable_sets
    - compute_feasible_sets

    All RA-based algorithms use a `SolverWrapper` for most of its computations.
    """

    def __init__(self, constraint_list, path, path_discretization, solver_wrapper='cvxpy'):
        """
        
        Parameters
        ----------
        constraint_list: 
        path: 
        path_discretization: 
        solver_wrapper: str, optional
            Name of the solver to use.
        """
        super(ReachabilityAlgorithm, self).__init__(constraint_list, path, path_discretization)
        if solver_wrapper=='cvxpy':
            self.solver_wrapper = cvxpyWrapper(self.constraints, self.path, self.path_discretization)
        else:
            self.solver_wrapper = cvxpyWrapper(self.constraints, self.path, self.path_discretization)

    def compute_feasible_sets(self):
        """ Return the set of feasible velocities along the path.

        Returns
        -------
        X: array, or list containing None
            Shape (N+1, 2). The tuple X[i] contains the lower and upper bound of the feasible
            set at s[i].  If there is no feasible state, X[i] is the tupele (None, None).

        """
        X = []
        for i in range(self.N + 1):
            H = np.zeros
            self.solver_wrapper.solve_stagewise_optim()

        
        
