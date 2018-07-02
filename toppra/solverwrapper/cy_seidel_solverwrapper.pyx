import numpy as np
cimport numpy as np
from libc.math cimport abs, pow, isnan
cimport cython

ctypedef np.int_t INT_t
ctypedef np.float_t FLOAT_t

from ..constraint import ConstraintType

cdef double TINY = 1e-20
cdef double VAR_MIN = -100000000
cdef double VAR_MAX =  100000000
cdef inline double max(double a, double b): return a if a > b else b
cdef inline double min(double a, double b): return a if a < b else b



cdef struct LpSol:
    # A struct that contains a solution to a Linear Program computed
    # using codes in this source file.
    int result
    double optval
    double[2] optvar
    int[2] active_c


def solve_lp1d(double[:] v, a, b, double low, double high):
    """Solve a Linear Program with 1 variable.

    See func:`cy_solve_lp1d` for more details.

    NOTE: Here a, b, are not typed because a valid input is the empty
    list. However, empty lists are not valid inputs to the cython
    version of this function. So, take extra precaution. 

    Another related note is by not statically typing `a` and `b`, this
    function is much slower than what it can be. My current benchmark
    for 1000 inequalities is 9ms, two-times slower than when there is
    static typing.

    """
    if a == [] or a is None:
        data = cy_solve_lp1d(v, 0, None, None, low, high)
    else:
        data = cy_solve_lp1d(v, len(a), a, b, low, high)
    return data.result, data.optval, data.optvar[0], data.active_c[0]


def solve_lp2d(double[:] v, double[:] a, double[:] b, double[:] c, double[:] low, double[:] high, INT_t[:] active_c):
    """ Solve a Linear Program with 2 variable.

    See func:`cy_solve_lp2d` for more details.
    """
    data = cy_solve_lp2d(v, a, b, c, low, high, active_c)
    return data.result, data.optval, data.optvar, data.active_c


@cython.boundscheck(True)
@cython.wraparound(True)
cdef LpSol cy_solve_lp1d(double[:] v, int nrows, double[:] a, double[:] b, double low, double high) except *:
    """ Solve the following program
    
            max   v[0] x + v[1]
            s.t.  a x <= b
                  low <= x <= high

    """
    cdef:
        double cur_min = low
        int active_c_min = -1
        double cur_max = high
        int active_c_max = -2
        unsigned int i
        double cur_x, optvar, optval
        LpSol solution
        double low_1d = VAR_MIN
        double high_1d = VAR_MAX

    for i in range(nrows):
        if a[i] > TINY:
            cur_x = - b[i] / a[i]
            if cur_x < cur_max:
                cur_max = cur_x
                active_c_max = i
        elif a[i] < -TINY:
            cur_x = - b[i] / a[i]
            if cur_x > cur_min:
                cur_min = cur_x
                active_c_min = i
        else:
            # a[i] is approximately zero. do nothing.
            pass

    if cur_min > cur_max:
        solution.result = 0
        return solution

    if abs(v[0]) < TINY or v[0] < 0:
        # optimizing direction is perpencicular to the line, or is
        # pointing toward the negative direction.
        solution.result = 1
        solution.optvar[0] = cur_min
        solution.optval = v[0] * cur_min + v[1]
        solution.active_c[0] = active_c_min
    else:
        solution.result = 1
        solution.optvar[0] = cur_max
        solution.optval = v[0] * cur_max + v[1]
        solution.active_c[0] = active_c_max

    return solution


@cython.boundscheck(True)
@cython.wraparound(True)
cdef LpSol cy_solve_lp2d(double[:] v, double[:] a, double[:] b, double[:] c, double[:] low, double[:] high, INT_t[:] active_c) except *:
    """ Solve a LP with two variables.

    The LP is specified as follow:
        max    v^T [x 1]
        s.t.   a x[0] + b x[1] + c <= 0
               low <= x <= high

    Parameters
    ----------
    v: double memoryview
    a: double memoryview
    b: double memoryview
    c: double memoryview
    low: double memoryview
    high: double memoryview
    active_c: int memoryview
        Contains (2) indicies of rows in a, b, c that are likely the
        active constraints at the optimal solution.

    Returns
    -------
    solution: LpSol
        A struct that contains the ouput. In the case of
        infeasibility, the struct has its result field equals 0. Other
        fields are not meaningful.

    """
    cdef:
        unsigned int nrows = a.shape[0]
        INT_t[:] index_map = np.arange(nrows)
        unsigned int i, k, j, n_wsr
        double[2] cur_optvar, zero_prj
        double[2] d_tan  # vector parallel to the line
        double[:] v_1d = np.zeros(2)  # optimizing direction
        double[:] a_1d = np.zeros(nrows + 4)
        double[:] b_1d = np.zeros(nrows + 4)
        double aj, bj, cj  # temporary symbols
        double low_1d = VAR_MIN
        double high_1d = VAR_MAX
        LpSol sol, sol_1d

    n_wsr = 0  # number of working set recomputation

    # If active_c contains valid entries, swap the first two indices
    # in index_map to these values.
    if active_c[0] >= 0 and active_c[0] < nrows and active_c[1] >= 0 and active_c[1] < nrows:
        index_map[0] = active_c[1]
        index_map[active_c[1]] = 0
        index_map[1] = active_c[0]
        index_map[active_c[0]] = 1

    # handle fixed bounds (low, high). The following convention is
    # adhered to: fixed bounds are assigned the numbers: -1, -2, -3,
    # -4 according to the following order: low[0], high[0], low[1],
    # high[1].
    for i in range(2):
        if v[i] > TINY:
            cur_optvar[i] = high[i]
            if i == 0:
                sol.active_c[0] = -2
            else:
                sol.active_c[1] = -4
        else:
            cur_optvar[i] = low[i]
            if i == 0:
                sol.active_c[0] = -1
            else:
                sol.active_c[1] = -3

    # handle other constraints in a, b, c
    for k in range(nrows):
        i = index_map[k]
        # if current optimal variable satisfies the i-th constraint, continue
        if a[i] * cur_optvar[0] + b[i] * cur_optvar[1] + c[i] < TINY:
            continue
        # otherwise, project all constraints on the line defined by (a[i], b[i], c[i])
        n_wsr += 1
        sol.active_c[0] = i
        # project the origin (0, 0) onto the new constraint
        # let ax + by + c=0 b the new constraint
        # let zero_prj be the projected point, one has
        #     zero_prj =  1 / (a^2 + b^2) [a  -b] [-c]
        #                              [b   a] [ 0]
        # this can be derived using perpendicularity
        # more specifically
        # zero_prj[0] = -ac / (a^2 + b^2), zero_prj[1] = -bc / (a^2 + b^2)
        zero_prj[0] = - a[i] * c[i] / (pow(a[i], 2) + pow(b[i], 2))
        zero_prj[1] = - b[i] * c[i] / (pow(a[i], 2) + pow(b[i], 2))
        d_tan[0] = -b[i]
        d_tan[1] = a[i]
        v_1d[0] = d_tan[0] * v[0] + d_tan[1] * v[1]
        v_1d[1] = 0
        # project 4 + k constraints onto the parallel line. each
        # constraint occupies a row on vectors a_1d, b_1d.
        nrows_1d = 4 + k
        for j in range(nrows_1d):
            # handle low <= x
            if (j == k):
                aj = -1
                bj = 0
                cj = low[0]
            # handle x <= high
            elif (j == k + 1):
                aj = 1
                bj = 0
                cj = -high[0]
            # handle low <= y 
            elif (j == k + 2):
                aj = 0
                bj = -1
                cj = low[1]
            # handle y <= high
            elif (j == k + 3):
                aj = 0
                bj = 1
                cj = -high[1]
            # handle other constraint
            else:
                aj = a[index_map[j]];
                bj = b[index_map[j]];
                cj = c[index_map[j]];
       
            # projective coefficients to the line
            denom = d_tan[0] * aj + d_tan[1] * bj

            # add respective coefficients to a_1d and b_1d
            if denom > TINY:
                t_limit = - (cj + zero_prj[1] * bj + zero_prj[0] * aj) / denom
                a_1d[j] = 1.0
                b_1d[j] = - t_limit
            elif denom < -TINY:
                t_limit = - (cj + zero_prj[1] * bj + zero_prj[0] * aj) / denom
                a_1d[j] = -1.0
                b_1d[j] = t_limit
            else:
                # the curretly considered constraint is parallel to
                # the base one. Check if they are infeasible, in which
                # case return failure immediately.
                if cj + zero_prj[1] * bj + zero_prj[0] * aj > 0:
                    sol.result = 0
                    return sol
                # feasible constraints, specify 0 <= 1
                a_1d[j] = 0
                b_1d[j] = 1.0

        # solve the projected, 1 dimensional LP
        sol_1d = cy_solve_lp1d(v_1d, nrows_1d, a_1d, b_1d, low_1d, high_1d)
        
        # 1d lp infeasible
        if sol_1d.result == 0:
            sol.result = 0
            return sol
        # feasible, wrapping up
        else:
            # compute the current optimal solution
            cur_optvar[0] = zero_prj[0] + sol_1d.optvar[0] * d_tan[0]
            cur_optvar[1] = zero_prj[1] + sol_1d.optvar[0] * d_tan[1]
            # record the active constraint's index
            if sol_1d.active_c[0] < k:
                sol.active_c[1] = index_map[sol_1d.active_c[0]]
            elif sol_1d.active_c[0] == k:
                sol.active_c[1] = -1
            elif sol_1d.active_c[0] == k + 1:
                sol.active_c[1] = -2
            elif sol_1d.active_c[0] == k + 2:
                sol.active_c[1] = -3
            elif sol_1d.active_c[0] == k + 3:
                sol.active_c[1] = -4
            else:
                assert False

    # Fill the solution struct
    sol.result = 1
    sol.optvar[0] = cur_optvar[0]
    sol.optvar[1] = cur_optvar[1]
    sol.optval = sol.optvar[0] * v[0] + sol.optvar[1] * v[1] + v[2]
    return sol

cdef class seidelWrapper:
    cdef:
        list constraints, _params
        unsigned int N, nV, nC, nCons
        double [:] path_discretization
        double [:] deltas
        double [:] a, b, c, low, high
        object path

        
    def __init__(self, list constraint_list, path, path_discretization):
        self.constraints = constraint_list
        self.path = path
        path_discretization = np.array(path_discretization)
        self.path_discretization = path_discretization
        self.N = len(path_discretization) - 1  # Number of stages. Number of point is _N + 1
        self.deltas = path_discretization[1:] - path_discretization[:-1]
        # safety checks
        assert path.get_path_interval()[0] == path_discretization[0]
        assert path.get_path_interval()[1] == path_discretization[-1]
        for i in range(self.N):
            assert path_discretization[i + 1] > path_discretization[i]

        # handle constraint parameters
        nCons = len(constraint_list)
        self.nCons = nCons
        self.nC = 2
        self.nV = 2
        self._params = []
        for i in range(nCons):
            if self.constraints[i].get_constraint_type() != ConstraintType.CanonicalLinear:
                raise NotImplementedError
            a, b, c, F, v, ubnd, xbnd = self.constraints[i].compute_constraint_params(
            self.path, self.path_discretization)
            if a is not None:
                self.nC += F.shape[1]
            self._params.append((a, b, c, F, v, ubnd, xbnd))

    @property
    def params(self):
        return self._params

    cpdef np.ndarray solve_stagewise_optim(self, unsigned int i, H, np.ndarray g, double x_min, double x_max, double x_next_min, double x_next_max):
        """Solve a stage-wise quadratic optimization.

        Parameters
        ----------
        i: int
            The stage index. See notes for details on each variable.
        H: array or None
        g: (2,)array
        x_min: float or nan
        x_max: float or nan
        x_next_min: float or nan
        x_next_max: float or nan

        Returns
        -------
        array
             If the optimization successes, return an array containing the optimal variable.
             Otherwise, the return array contains NaN (numpy.nan).
        """
        assert i <= self.N and 0 <= i

        # fill coefficient
        cdef:
            np.ndarray a = np.zeros(self.nC)
            np.ndarray b = np.array(a)
            np.ndarray c = np.array(a)
            double [2] low, high
            INT_t [2] active_c
            int cur_index = 0, j, nC
            LpSol solution
        low[:] = [VAR_MIN, VAR_MIN]
        high[:] = [VAR_MAX, VAR_MAX]
        active_c[:] = [0, 1]

        # handle x_min <= x_i <= x_max
        if not isnan(x_min):
            low[1] = max(low[1], x_min)
        if not isnan(x_max):
            high[1] = min(high[1], x_max)

        # handle x_next_min <= 2 delta u + x_i <= x_next_max
        if i < self.N:
            if isnan(x_next_min):
                c[0] = -1
            else:
                a[0] = - 2 * self.deltas[i]
                b[0] = - 1.0
                c[0] = x_next_min
            if isnan(x_next_max):
                c[1] = -1
            else:
                a[1] = 2 * self.deltas[i]
                b[1] = 1.0
                c[1] = - x_next_max
        else:
            # at last stage, do not consider this constraint
            c[0] = -1
            c[1] = -1

        # handle constraint from the parameters
        cur_index = 2
        for j in range(self.nCons):
            a_j, b_j, c_j, F_j, v_j, ubound_j, xbound_j = self._params[j]
            
            if a_j is not None:
                nC_ = F_j[i].shape[0]
                a[cur_index: cur_index + nC_] = F_j[i].dot(a_j[i])
                b[cur_index: cur_index + nC_] = F_j[i].dot(b_j[i])
                c[cur_index: cur_index + nC_] = - v_j[i] + F_j[i].dot(c_j[i])
                cur_index = cur_index + nC_

            if ubound_j is not None:
                low[0] = max(low[0], ubound_j[i, 0])
                high[0] = min(high[0], ubound_j[i, 1])

            if xbound_j is not None:
                low[1] = max(low[1], xbound_j[i, 0])
                high[1] = min(high[1], xbound_j[i, 1])

        # solve the lp
        v = np.zeros(3)
        v[:2] = -g

        solution = cy_solve_lp2d(v, a, b, c, low, high, active_c)

        if solution.result == 0:
            var = np.zeros(2)
            var[0] = np.nan
            var[1] = np.nan
        else:
            var = np.asarray(solution.optvar)
        return var

    def setup_solver(self):
        pass

    def close_solver(self):
        pass

