from .solverwrapper import SolverWrapper
import numpy as np
from ..constraint import ConstraintType
from ..constants import INFTY

try:
    from qpoases import (
        PyOptions as Options,
        PyPrintLevel as PrintLevel,
        PyReturnValue as ReturnValue,
        PySQProblem as SQProblem,
    )

    qpoases_FOUND = True
except ImportError:
    qpoases_FOUND = False

INF = INFTY


class qpOASESSolverWrapper(SolverWrapper):
    """ A solver wrapper using `qpOASES`.

    Parameters
    ----------
    constraint_list: list of :class:`.Constraint`
        The constraints the robot is subjected to.
    path: :class:`.Interpolator`
        The geometric path.
    path_discretization: array
        The discretized path positions.
    """

    def __init__(self, constraint_list, path, path_discretization):
        assert qpoases_FOUND, "toppra is unable to find any installation of qpoases!"
        super(qpOASESSolverWrapper, self).__init__(
            constraint_list, path, path_discretization
        )

        # Currently only support Canonical Linear Constraint
        self.nC = 2  # First constraint is x + 2 D u <= xnext_max, second is xnext_min <= x + 2D u
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
        self._lA = -np.ones(self.nC)
        self._hA = -np.ones(self.nC)

        option = Options()
        option.printLevel = PrintLevel.NONE
        self.solver = SQProblem(self.nV, self.nC)
        self.solver.setOptions(option)

    def solve_stagewise_optim(self, i, H, g, x_min, x_max, x_next_min, x_next_max):
        # NOTE: qpOASES solve QPs of the following form:
        #  min    0.5 y^T H y + g^T y
        #  s.t    lA <= Ay <= hA
        #         l  <=  y <= h
        assert i <= self.N and 0 <= i

        l = -np.ones(2) * INF
        h = np.ones(2) * INF

        if x_min is not None:
            l[1] = max(l[1], x_min)
        if x_max is not None:
            h[1] = min(h[1], x_max)

        if i < self.N:
            delta = self.get_deltas()[i]
            if x_next_min is not None:
                self._A[0] = [-2 * delta, -1]
                self._hA[0] = -x_next_min
                self._lA[0] = -INF
            else:
                self._A[0] = [0, 0]
                self._hA[0] = INF
                self._lA[0] = -INF
            if x_next_max is not None:
                self._A[1] = [2 * delta, 1]
                self._hA[1] = x_next_max
                self._lA[1] = -INF
            else:
                self._A[1] = [0, 0]
                self._hA[1] = INF
                self._lA[1] = -INF
        cur_index = 2
        for j in range(len(self.constraints)):
            a, b, c, F, v, ubound, xbound = self.params[j]

            # Case 1
            if a is not None:
                if self.constraints[j].identical:
                    nC_ = F.shape[0]
                    self._A[cur_index : cur_index + nC_, 0] = F.dot(a[i])
                    self._A[cur_index : cur_index + nC_, 1] = F.dot(b[i])
                    self._hA[cur_index : cur_index + nC_] = v - F.dot(c[i])
                    self._lA[cur_index : cur_index + nC_] = -INF
                else:
                    nC_ = F[i].shape[0]
                    self._A[cur_index : cur_index + nC_, 0] = F[i].dot(a[i])
                    self._A[cur_index : cur_index + nC_, 1] = F[i].dot(b[i])
                    self._hA[cur_index : cur_index + nC_] = v[i] - F[i].dot(c[i])
                    self._lA[cur_index : cur_index + nC_] = -INF
                cur_index = cur_index + nC_

            if ubound is not None:
                l[0] = max(l[0], ubound[i, 0])
                h[0] = min(h[0], ubound[i, 1])

            if xbound is not None:
                l[1] = max(l[1], xbound[i, 0])
                h[1] = min(h[1], xbound[i, 1])

        if H is None:
            H = np.zeros((self.get_no_vars(), self.get_no_vars()))

        res = self.solver.init(
            H, g, self._A, l, h, self._lA, self._hA, np.array([1000])
        )
        if res == ReturnValue.SUCCESSFUL_RETURN:
            var = np.zeros(self.nV)
            self.solver.getPrimalSolution(var)
            return var
        else:
            res = np.empty(self.get_no_vars())
            res[:] = np.nan
            return res
