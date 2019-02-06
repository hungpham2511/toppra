from .solverwrapper import SolverWrapper
import numpy as np
from ..constraint import ConstraintType
from ..constants import QPOASES_INFTY

try:
    from qpoases import (PyOptions as Options, PyPrintLevel as PrintLevel,
                         PyReturnValue as ReturnValue, PySQProblem as SQProblem)
    qpoases_FOUND = True
except ImportError:
    qpoases_FOUND = False
import logging

logger = logging.getLogger(__name__)

eps = 1e-8  # Coefficient to check for qpoases tolerances TODO: shift this to constants.py


class hotqpOASESSolverWrapper(SolverWrapper):
    """A solver wrapper using `qpOASES`.

    This wrapper takes advantage of the warm-start capability of the
    qpOASES quadratic programming solver. It uses two different
    qp solvers. One to solve for maximized controllable sets and one to
    solve for minimized controllable sets. The wrapper selects which solver
    to use by looking at the optimization direction.

    If the logger "toppra" is set to debug level, qpoases solvers are
    initialized with PrintLevel.HIGH. Otherwise, these are initialized
    with PrintLevel.NONE

    Currently only support Canonical Linear Constraints.

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
    scaling_solverwrapper: bool, optional
        If is True, try to scale the data of each optimization before running.
    """
    def __init__(self, constraint_list, path, path_discretization, disable_check=False, scaling_solverwrapper=True):
        assert qpoases_FOUND, "toppra is unable to find any installation of qpoases!"
        super(hotqpOASESSolverWrapper, self).__init__(constraint_list, path, path_discretization)
        self._disable_check = disable_check

        # First constraint is x + 2 D u <= xnext_max, second is xnext_min <= x + 2D u
        self.nC = 2  # number of Constraints.
        for i, constraint in enumerate(constraint_list):
            if constraint.get_constraint_type() != ConstraintType.CanonicalLinear:
                raise NotImplementedError
            a, b, c, F, v, _, _ = self.params[i]
            if a is not None:
                if constraint.identical:
                    self.nC += F.shape[0]
                else:
                    self.nC += F.shape[1]

        # qpOASES coefficient arrays
        # l <= var <= h
        # lA <= A var <= hA
        self._A = np.zeros((self.nC, self.nV))
        self._lA = - np.ones(self.nC) * QPOASES_INFTY
        self._hA = np.ones(self.nC) * QPOASES_INFTY
        self._l = - np.ones(2) * QPOASES_INFTY
        self._h = np.ones(2) * QPOASES_INFTY

    def setup_solver(self):
        option = Options()
        if logger.getEffectiveLevel() == logging.DEBUG:
            # option.printLevel = PrintLevel.HIGH
            option.printLevel = PrintLevel.NONE
        else:
            option.printLevel = PrintLevel.NONE
        self.solver_minimizing = SQProblem(self.nV, self.nC)
        self.solver_minimizing.setOptions(option)
        self.solver_maximizing = SQProblem(self.nV, self.nC)
        self.solver_maximizing.setOptions(option)

        self.solver_minimizing_recent_index = -2
        self.solver_maximizing_recent_index = -2

    def close_solver(self):
        self.solver_minimizing = None
        self.solver_maximizing = None

    def solve_stagewise_optim(self, i, H, g, x_min, x_max, x_next_min, x_next_max):
        # NOTE: qpOASES solve QPs of the following form:
        #  min    0.5 y^T H y + g^T y
        #  s.t    lA <= Ay <= hA
        #         l  <=  y <= h
        assert i <= self.N and 0 <= i

        self._l[:] = - QPOASES_INFTY
        self._h[:] = QPOASES_INFTY

        # variable scale
        # solve instead of the optimization
        #  min    0.5 y^T scale H scale y + g^T scale y
        #  s.t    lA <= A scale y <= hA
        #         l  <=  scale y <= h
        variable_scales = np.array([2.0, 2.0])
        variable_scales_mat = np.diag(variable_scales)

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
                self._hA[0] = QPOASES_INFTY
            if x_next_max is not None:
                self._A[1] = [2 * delta, 1]
                self._hA[1] = x_next_max
            else:
                self._A[1] = [0, 0]
                self._hA[1] = QPOASES_INFTY
        cur_index = 2
        for j in range(len(self.constraints)):
            a, b, c, F, v, ubound, xbound = self.params[j]

            if a is not None:
                if self.constraints[j].identical:
                    nC_ = F.shape[0]
                    self._A[cur_index: cur_index + nC_, 0] = F.dot(a[i])
                    self._A[cur_index: cur_index + nC_, 1] = F.dot(b[i])
                    self._hA[cur_index: cur_index + nC_] = v - F.dot(c[i])
                    self._lA[cur_index: cur_index + nC_] = - QPOASES_INFTY
                else:
                    nC_ = F[i].shape[0]
                    self._A[cur_index: cur_index + nC_, 0] = F[i].dot(a[i])
                    self._A[cur_index: cur_index + nC_, 1] = F[i].dot(b[i])
                    self._hA[cur_index: cur_index + nC_] = v[i] - F[i].dot(c[i])
                    self._lA[cur_index: cur_index + nC_] = - QPOASES_INFTY
                cur_index = cur_index + nC_
            if ubound is not None:
                self._l[0] = max(self._l[0], ubound[i, 0])
                self._h[0] = min(self._h[0], ubound[i, 1])

            if xbound is not None:
                self._l[1] = max(self._l[1], xbound[i, 0])
                self._h[1] = min(self._h[1], xbound[i, 1])

        if H is None:
            H = np.zeros((self.get_no_vars(), self.get_no_vars()))

        # check the ratio of A[:, 0] and A[:, 1], if this is too far
        # from 1, the problem is badly scaled.
        if logger.isEnabledFor(logging.DEBUG):
            A_abs = np.abs(self._A)
            ratios = np.mean(A_abs[:, 0] / A_abs[:, 1])
            logger.debug("Coefficient ratio: A[:, 0] / A[:, 1] = {:f}".format(ratios))

        # scaling
        self._A = self._A.dot(variable_scales_mat)
        self._l = self._l / variable_scales
        self._h = self._h / variable_scales
        g = g * variable_scales
        H = variable_scales_mat.dot(H).dot(variable_scales_mat)

        # Select what solver to use
        if g[1] > 0:  # Choose solver_minimizing
            if abs(self.solver_minimizing_recent_index - i) > 1:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("Choose solver [up] - init")
                res = self.solver_minimizing.init(H, g, self._A, self._l, self._h, self._lA, self._hA, np.array([1000]))
            else:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("Choose solver [up] - hotstart")
                res = self.solver_minimizing.hotstart(H, g, self._A, self._l, self._h, self._lA, self._hA, np.array([1000]))
            self.solver_minimizing_recent_index = i
        else:  # Choose solver_maximizing
            if abs(self.solver_maximizing_recent_index - i) > 1:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("Choose solver [down] - init")
                res = self.solver_maximizing.init(H, g, self._A, self._l, self._h, self._lA, self._hA, np.array([1000]))
            else:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("Choose solver [down] - hotstart")
                res = self.solver_maximizing.hotstart(H, g, self._A, self._l, self._h, self._lA, self._hA, np.array([1000]))
            self.solver_maximizing_recent_index = i

        if res == ReturnValue.SUCCESSFUL_RETURN:
            var = np.zeros(self.nV)
            if g[1] > 0:
                self.solver_minimizing.getPrimalSolution(var)
            else:
                self.solver_maximizing.getPrimalSolution(var)

            logger.debug("optimization value: {:}".format(var))

            if self._disable_check:
                return var * variable_scales

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
                return var * variable_scales
        else:
            logger.debug("Optimization fails. qpOASES error code: {:d}. Checking constraint feasibility for (0, 0)!".format(res))

            if (np.all(0 <= self._hA) and np.all(0 >= self._lA) and np.all(0 <= self._h) and np.all(0 >= self._l)):
                logger.fatal("(0, 0) satisfies all constraints => error due to numerical errors.")
                print(self._A)
            else:
                logger.debug("(0, 0) does not satisfy all constraints.")

        res = np.empty(self.get_no_vars())
        res[:] = np.nan
        return res
