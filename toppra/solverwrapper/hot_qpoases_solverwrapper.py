from .solverwrapper import SolverWrapper
import numpy as np
from ..constraint import ConstraintType
from ..constants import INFTY
try:
    from qpoases import (PyOptions as Options, PyPrintLevel as PrintLevel,
                         PyReturnValue as ReturnValue, PySQProblem as SQProblem)
    qpoases_FOUND = True
except ImportError:
    qpoases_FOUND = False
import logging
logger = logging.getLogger(__name__)


eps = 1e-8  # Coefficient to check for qpoases tolerances
INF = INFTY


class hotqpOASESSolverWrapper(SolverWrapper):
    """A solver wrapper using `qpOASES`.

    This wrapper takes advantage of the warm-start capability of
    qpOASES quadratic programming solver by using two different
    solvers. One to solve for maximized controllable sets and one to
    solve for minimized controllable sets.

    If the logger "toppra" is set to debug level, qpoases solvers are
    initialized with PrintLevel.HIGH. Otherwise, these are initialized
    with PrintLevel.NONE

    Parameters
    ----------
    constraint_list: list of :class:`.Constraint`
        The constraints the robot is subjected to.
    path: :class:`.Interpolator`
        The geometric path.
    path_discretization: array
        The discretized path positions.
    disable_check: bool, optional
        Disable check for solution validity. Improve speed by about
        20% but entails the possibility that failure is not reported
        correctly.

    """
    def __init__(self, constraint_list, path, path_discretization, disable_check=False):
        assert qpoases_FOUND, "toppra is unable to find any installation of qpoases!"
        super(hotqpOASESSolverWrapper, self).__init__(constraint_list, path, path_discretization)
        self._disable_check = disable_check
        # Currently only support Canonical Linear Constraint
        self.nC = 2 # First constraint is x + 2 D u <= xnext_max, second is xnext_min <= x + 2D u
        for i, constraint in enumerate(constraint_list):
            if constraint.get_constraint_type() != ConstraintType.CanonicalLinear:
                raise NotImplementedError
            a, b, c, F, v, _, _ = self.params[i]
            if a is not None:
                if constraint.identical:
                    self.nC += F.shape[0]
                else:
                    self.nC += F.shape[1]

        self._A = np.zeros((self.nC, self.nV))
        self._lA = - np.ones(self.nC) * INF
        self._hA = np.ones(self.nC) * INF
        self._l = - np.ones(2) * INF
        self._h = np.ones(2) * INF

    def setup_solver(self):
        option = Options()
        if logger.getEffectiveLevel() == logging.DEBUG:
            # option.printLevel = PrintLevel.HIGH
            option.printLevel = PrintLevel.NONE
        else:
            option.printLevel = PrintLevel.NONE
        self.solver_up = SQProblem(self.nV, self.nC)
        self.solver_up.setOptions(option)
        self.solver_down = SQProblem(self.nV, self.nC)
        self.solver_down.setOptions(option)

        self.solver_up_recent_index = -2
        self.solver_down_recent_index = -2

    def close_solver(self):
        self.solver_up = None
        self.solver_down = None

    def solve_stagewise_optim(self, i, H, g, x_min, x_max, x_next_min, x_next_max):
        # NOTE: qpOASES solve QPs of the following form:
        #  min    0.5 y^T H y + g^T y
        #  s.t    lA <= Ay <= hA
        #         l  <=  y <= h
        assert i <= self.N and 0 <= i

        self._l[:] = -INF
        self._h[:] = INF

        if x_min is not None:
            self._l[1] = max(self._l[1], x_min)
        if x_max is not None:
            self._h[1] = min(self._h[1], x_max)

        if i < self.N:
            delta = self.get_deltas()[i]
            if x_next_min is not None:
                self._A[0] = [-2 * delta, -1]
                self._hA[0] = - x_next_min
            else:
                self._A[0] = [0, 0]
                self._hA[0] = INF
            if x_next_max is not None:
                self._A[1] = [2 * delta, 1]
                self._hA[1] = x_next_max
            else:
                self._A[1] = [0, 0]
                self._hA[1] = INF
        cur_index = 2
        for j in range(len(self.constraints)):
            a, b, c, F, v, ubound, xbound = self.params[j]

            if a is not None:
                if self.constraints[j].identical:
                    nC_ = F.shape[0]
                    self._A[cur_index: cur_index + nC_, 0] = F.dot(a[i])
                    self._A[cur_index: cur_index + nC_, 1] = F.dot(b[i])
                    self._hA[cur_index: cur_index + nC_] = v - F.dot(c[i])
                    self._lA[cur_index: cur_index + nC_] = - INF
                else:
                    nC_ = F[i].shape[0]
                    self._A[cur_index: cur_index + nC_, 0] = F[i].dot(a[i])
                    self._A[cur_index: cur_index + nC_, 1] = F[i].dot(b[i])
                    self._hA[cur_index: cur_index + nC_] = v[i] - F[i].dot(c[i])
                    self._lA[cur_index: cur_index + nC_] = - INF
                cur_index = cur_index + nC_
            if ubound is not None:
                self._l[0] = max(self._l[0], ubound[i, 0])
                self._h[0] = min(self._h[0], ubound[i, 1])

            if xbound is not None:
                self._l[1] = max(self._l[1], xbound[i, 0])
                self._h[1] = min(self._h[1], xbound[i, 1])

        if H is None:
            H = np.zeros((self.get_no_vars(), self.get_no_vars()))

        # Select what solver to use
        if g[1] > 0:  # Choose solver_up
            if abs(self.solver_up_recent_index - i) > 1:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("Choose solver [up] - init")
                res = self.solver_up.init(H, g, self._A, self._l, self._h, self._lA, self._hA, np.array([1000]))
            else:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("Choose solver [up] - hotstart")
                res = self.solver_up.hotstart(H, g, self._A, self._l, self._h, self._lA, self._hA, np.array([1000]))
            self.solver_up_recent_index = i
        else:  # Choose solver_down
            if abs(self.solver_down_recent_index - i) > 1:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("Choose solver [down] - init")
                res = self.solver_down.init(H, g, self._A, self._l, self._h, self._lA, self._hA, np.array([1000]))
            else:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("Choose solver [down] - hotstart")
                res = self.solver_down.hotstart(H, g, self._A, self._l, self._h, self._lA, self._hA, np.array([1000]))
            self.solver_down_recent_index = i

        if res == ReturnValue.SUCCESSFUL_RETURN:
            var = np.zeros(self.nV)
            if g[1] > 0:
                self.solver_up.getPrimalSolution(var)
            else:
                self.solver_down.getPrimalSolution(var)

            if self._disable_check:
                return var

            # Check for constraint feasibility
            success = (np.all(self._l <= var + eps) and np.all(var <= self._h + eps)
                       and np.all(np.dot(self._A, var) <= self._hA + eps)
                       and np.all(np.dot(self._A, var) >= self._lA - eps))
            if not success:
                # import ipdb; ipdb.set_trace()
                logger.fatal("Hotstart fails but qpOASES does not report correctly. \n "
                             "var: {:}, lower_bound: {:}, higher_bound{:}".format(var, self._l, self._h))
                # TODO: Investigate why this happen and fix the
                # relevant code (in qpOASES wrapper)
            else:
                return var

        res = np.empty(self.get_no_vars())
        res[:] = np.nan
        return res
