# cython: embedsignature=True
import numpy as np
cimport numpy as np
from libc.math cimport abs, pow, isnan
cimport cython
from cpython.array cimport array, clone

ctypedef np.int_t INT_t
ctypedef np.float_t FLOAT_t

from ..constraint import ConstraintType

cdef inline double dbl_max(double a, double b): return a if a > b else b
cdef inline double dbl_min(double a, double b): return a if a < b else b

# constants
cdef double TINY = 1e-10
cdef double SMALL = 1e-8

# bounds on variable used in seidel solver wrapper. u and x at every
# stage is constrained to stay within this range.
cdef double VAR_MIN = -100000000  
cdef double VAR_MAX =  100000000

# absolute largest value that a variable can have. This bound should
# be never be reached, however, in order for the code to work properly.
cdef double INF = 10000000000  

cdef double NAN = float("NaN")



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
    if a is None:
        data = cy_solve_lp1d(v, 0, None, None, low, high)
    elif len(a) == 0:
        data = cy_solve_lp1d(v, 0, None, None, low, high)
    else:
        data = cy_solve_lp1d(v, len(a), a, b, low, high)
    return data.result, data.optval, data.optvar[0], data.active_c[0]

def solve_lp2d(double[:] v, double[:] a, double[:] b, double[:] c, double[:] low, double[:] high, INT_t[:] active_c):
    """ Solve a Linear Program with 2 variable.

    See func:`cy_solve_lp2d` for more details.
    """
    cdef:
        INT_t [:] idx_map
        double[:] a_1d
        double[:] b_1d
    if a is not None and len(a) != 0:
        nrows = a.shape[0]
        idx_map = np.zeros_like(a, dtype=int)
        a_1d = np.zeros(nrows + 4)
        b_1d = np.zeros(nrows + 4)
        use_cache = True
    else:
        idx_map = None
        a_1d = None
        b_1d = None
        use_cache = False

    data = cy_solve_lp2d(v, a, b, c, low, high, active_c, use_cache, idx_map, a_1d, b_1d)
    return data.result, data.optval, data.optvar, data.active_c


# @cython.profile(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef LpSol cy_solve_lp1d(double[:] v, int nrows, double[:] a, double[:] b, double low, double high) except *:
    """ Solve the following program
    
            dbl_max   v[0] x + v[1]
            s.t.  a x + b <= 0
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

# @cython.profile(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef LpSol cy_solve_lp2d(double[:] v, double[:] a, double[:] b, double[:] c,
                         double[:] low, double[:] high,
                         INT_t[:] active_c, bint use_cache,
                         INT_t [:] index_map, double[:] a_1d, double[:] b_1d) except *:
    """ Solve a LP with two variables.

    The LP is specified as follow:
         dbl_max    v^T [x 1]
            s.t.    a x[0] + b x[1] + c <= 0
                    low <= x <= high

    NOTE: A possible optimization for this function is pruning linear
    constraints that are clearly infeasible. This is not implemented
    because in my current code, the bottleneck is not in solving
    TOPP-RA but in setting up the parameters.

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
    use_cache: bool
    index_map: int memoryview
        A view to a pre-allocated integer array, to map from
        [1,...,nrows] to the considered entries. This array is created
        to avoid the cost of initializing a new array.
    a_1d: double memoryview
        A view to an initialized array. This array is created to avoid
        the cost of initializing a new array.
    b_1d: double memoryview
        A view to an initialized array. This array is created to avoid
        the cost of initializing a new array.

    Returns
    -------
    solution: LpSol
        A struct that contains the ouput. In the case of
        infeasibility, the struct has its result field equals 0. Other
        fields are not meaningful.

    """
    cdef:
        # number of working set recomputation
        unsigned int nrows = a.shape[0], nrows_1d, n_wsr = 0  
        unsigned int i, k, j
        double[2] cur_optvar, zero_prj
        double[2] d_tan  # vector parallel to the line
        double[2] v_1d_  # optimizing direction
        double [:] v_1d = v_1d_
        double aj, bj, cj  # temporary symbols

        # absolute bounds used in solving the 1 dimensional
        # optimization sub-problems. These bounds needs to be very
        # large, so that they are never active at the optimization
        # solution of these 1D subproblem..
        double low_1d = - INF  
        double high_1d = INF
        LpSol sol, sol_1d

    v_1d_[:] = [0, 0]
    # print all input to the algorithm
    # print "v={:}\n a={:}\n b={:}\n c={:}\n low={:}\n high={:}\n active_c {:}".format(
    #     *map(repr, map(np.asarray,
    #                    [v, a, b, c, low, high, active_c])))

    if not use_cache:
        index_map = np.arange(nrows)
        v_1d = np.zeros(2)  # optimizing direction
        a_1d = np.zeros(nrows + 4)
        b_1d = np.zeros(nrows + 4)
    else:
        assert index_map.shape[0] == nrows

    # handle fixed bounds (low, high). The following convention is
    # adhered to: fixed bounds are assigned the numbers: -1, -2, -3,
    # -4 according to the following order: low[0], high[0], low[1],
    # high[1].
    for i in range(2):
        if low[i] > high[i]:
            sol.result = 0
            return sol
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
    # print("v: {:}".format(repr(np.array(v))))
    # print("opt_var: {:}".format(repr(np.array(cur_optvar))))

    # If active_c contains valid entries, swap the first two indices
    # in index_map to these values.
    cdef unsigned int cur_row = 2
    if active_c[0] >= 0 and active_c[0] < nrows and active_c[1] >= 0 and active_c[1] < nrows and active_c[0] != active_c[1]:
        # active_c contains valid indices
        index_map[0] = active_c[1]
        index_map[1] = active_c[0]
        for i in range(nrows):
            if i != active_c[0] and i != active_c[1]:
                index_map[cur_row] = i
                cur_row += 1
    else:
        for i in range(nrows):
            index_map[i] = i

    # pre-process the inequalities, remove those that are redundant
    cdef cloned_index_map 
    # print(np.array(index_map))

    # handle other constraints in a, b, c
    for k in range(nrows):
        i = index_map[k]
        # if current optimal variable satisfies the i-th constraint, continue
        if a[i] * cur_optvar[0] + b[i] * cur_optvar[1] + c[i] < TINY:
            continue
        # print("a[i] * cur_optvar[0] + b[i] * cur_optvar[1] + c[i] = {:f}".format(
        #     a[i] * cur_optvar[0] + b[i] * cur_optvar[1] + c[i]))
        # print k
        # otherwise, project all constraints on the line defined by (a[i], b[i], c[i])
        n_wsr += 1
        sol.active_c[0] = i
        # project the origin (0, 0) onto the new constraint
        # let ax + by + c=0 b the new constraint
        # let zero_prj be the projected point, one has
        #     zero_prj =  1 / (a^2 + b^2) [a  -b] [-c]
        #                                 [b   a] [ 0]
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
                if cj + zero_prj[1] * bj + zero_prj[0] * aj > SMALL:
                    sol.result = 0
                    return sol
                # feasible constraints, specify 0 <= 1
                a_1d[j] = 0
                b_1d[j] = - 1.0

        # solve the projected, 1 dimensional LP
        sol_1d = cy_solve_lp1d(v_1d, nrows_1d, a_1d, b_1d, low_1d, high_1d)
        
        # 1d lp infeasible
        if sol_1d.result == 0:
            sol.result = 0
            return sol
        # feasible, wrapping up
        else:
            # print "v={:}\n a={:}\n b={:}\n low={:}\n high={:}\n nrows={:}".format(
            #     *map(repr, map(np.asarray,
            #                    [v_1d, a_1d, b_1d, low_1d, high_1d, nrows_1d])))
            # compute the current optimal solution
            cur_optvar[0] = zero_prj[0] + sol_1d.optvar[0] * d_tan[0]
            cur_optvar[1] = zero_prj[1] + sol_1d.optvar[0] * d_tan[1]
            # print("opt_var: {:}".format(repr(np.array(cur_optvar))))
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
                # the algorithm should not reach this point. If it
                # does, this means the active constraint at the
                # optimal solution is the fixed bound used in the 1
                # dimensional subproblem. This should not happen
                # though.
                sol.result = 0
                return sol

    # Fill the solution struct
    sol.result = 1
    sol.optvar[0] = cur_optvar[0]
    sol.optvar[1] = cur_optvar[1]
    sol.optval = sol.optvar[0] * v[0] + sol.optvar[1] * v[1] + v[2]
    return sol

cdef class seidelWrapper:
    """ A solver wrapper that implements Seidel's LP algorithm.

    This wrapper can only be used if there is only Canonical Linear
    Constraints.

    Parameters
    ----------
    constraint_list: list of :class:`.Constraint`
        The constraints the robot is subjected to.
    path: :class:`.Interpolator`
        The geometric path.
    path_discretization: array
        The discretized path positions.
    """
    cdef:
        list constraints, _params
        unsigned int N, nV, nC, nCons
        double [:] path_discretization
        double [:] deltas
        double [:] a, b, c, low, high, a_1d, b_1d, v
        object path
        INT_t [:] active_c_up, active_c_down
        INT_t [:] index_map
        double [:, ::1] a_arr, b_arr, c_arr  # mmviews of coefficients of the 2D Lp
        double [:, :] low_arr, high_arr    # mmviews of coefficients of the 2D Lp
        double scaling # path scaling
        
    # @cython.profile(True)
    def __init__(self, list constraint_list, path, path_discretization):
        """ A solver wrapper that implements Seidel's LP algorithm.

        This wrapper can only be used if there is only Canonical Linear
        Constraints.

        Parameters
        ----------
        constraint_list: list of :class:`.Constraint`
            The constraints the robot is subjected to.
        path: :class:`.Interpolator`
            The geometric path.
        path_discretization: array
            The discretized path positions.
        """
        self.constraints = constraint_list
        self.path = path
        path_discretization = np.array(path_discretization)
        self.path_discretization = path_discretization
        self.N = len(path_discretization) - 1  # Number of stages. Number of point is _N + 1
        self.deltas = path_discretization[1:] - path_discretization[:-1]
        cdef unsigned int cur_index, j, i, k

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
                self.path, path_discretization)
            if a is not None:
                if self.constraints[i].identical:
                    self.nC += F.shape[0]
                else:
                    self.nC += F.shape[1]
            self._params.append((a, b, c, F, v, ubnd, xbnd))

        # init constraint coefficients for the 2d lps, which are
        # self.a_arr, self.b_arr, self.c_arr, self.low_arr,
        # self.high_arr. The first dimensions of these array all equal
        # N + 1.
        self.a_arr = np.zeros((self.N + 1, self.nC))
        self.b_arr = np.zeros((self.N + 1, self.nC))
        self.c_arr = np.zeros((self.N + 1, self.nC))
        self.low_arr = np.ones((self.N + 1, 2)) * VAR_MIN
        self.high_arr = np.ones((self.N + 1, 2)) * VAR_MAX
        cur_index = 2

        cdef double [:, :] ta, tb, tc
        cdef unsigned int nC_
        for j in range(nCons):
            a_j, b_j, c_j, F_j, h_j, ubound_j, xbound_j = self._params[j]

            if a_j is not None:
                if self.constraints[j].identical:
                    nC_ = F_j.shape[0]
                    # <- Most time consuming code, but this computation seems unavoidable
                    ta = a_j.dot(F_j.T)
                    tb = b_j.dot(F_j.T)
                    tc = c_j.dot(F_j.T) - h_j
                    # <-- End

                    for i in range(self.N + 1):
                        for k in range(nC_):
                            self.a_arr[i, cur_index + k] = ta[i, k]
                            self.b_arr[i, cur_index + k] = tb[i, k]
                            self.c_arr[i, cur_index + k] = tc[i, k]
                else:
                    nC_ = F_j.shape[1]
                    for i in range(self.N + 1):
                        tai = np.dot(F_j[i], a_j[i])
                        tbi = np.dot(F_j[i], b_j[i])
                        tci = np.dot(F_j[i], c_j[i]) - h_j[i]
                        for k in range(nC_):
                            self.a_arr[i, cur_index + k] = tai[k]
                            self.b_arr[i, cur_index + k] = tbi[k]
                            self.c_arr[i, cur_index + k] = tci[k]
                cur_index += nC_

            if ubound_j is not None:
                for i in range(self.N + 1):
                    self.low_arr[i, 0] = dbl_max(self.low_arr[i, 0], ubound_j[i, 0])
                    self.high_arr[i, 0] = dbl_min(self.high_arr[i, 0], ubound_j[i, 1])

            if xbound_j is not None:
                for i in range(self.N + 1):
                    self.low_arr[i, 1] = dbl_max(self.low_arr[i, 1], xbound_j[i, 0])
                    self.high_arr[i, 1] = dbl_min(self.high_arr[i, 1], xbound_j[i, 1])
                    
        # init constraint coefficients for the 1d LPs
        self.a_1d = np.zeros(self.nC + 4)
        self.b_1d = np.zeros(self.nC + 4)
        self.index_map = np.zeros(self.nC, dtype=int)
        self.active_c_up = np.zeros(2, dtype=int)
        self.active_c_down = np.zeros(2, dtype=int)
        self.v = np.zeros(3)

    def get_no_vars(self):
        return self.nV

    def get_no_stages(self):
        return self.N

    def get_deltas(self):
        return np.asarray(self.deltas)

    @property
    def params(self):
        return self._params

    # @cython.profile(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef solve_stagewise_optim(self, unsigned int i, H, np.ndarray g, double x_min, double x_max, double x_next_min, double x_next_max):
        """Solve a stage-wise linear optimization problem.

        The linear optimization problem is described below.

        .. math::
            \\text{min  }  & [u, x] g    \\\\
            \\text{s.t.  } & [u, x] \\text{ is feasible at stage } i \\\\
                           & x_{min} \leq x \leq x_{max}             \\\\
                           & x_{next, min} \leq x + 2 \Delta_i u \leq x_{next, max},

        TODO if x_min == x_max, one can solve an LP instead of a 2D
        LP. This optimization is currently not implemented.

        Parameters
        ----------
        i: int
            The stage index. See notes for details on each variable.
        H: array or None
            This term is not used and is neglected.
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
        double C array or list
             If successes, return an array containing the optimal
             variable.  Since NaN is also a valid double, this list
             contains NaN if the optimization problem is infeasible.
        """
        assert i <= self.N and 0 <= i

        # fill coefficient
        cdef:
            unsigned int k, cur_index = 0, j, nC  # indices
            double [2] var  # Result
            LpSol  # Solution struct to hold the 2D or 1D LP result
            double low_arr[2], high_arr[2]
        low_arr[0] = self.low_arr[i, 0]
        low_arr[1] = self.low_arr[i, 1]
        high_arr[0] = self.high_arr[i, 0]
        high_arr[1] = self.high_arr[i, 1]

        # handle x_min <= x_i <= x_max
        if not isnan(x_min):
            low_arr[1] = dbl_max(low_arr[1], x_min)
        if not isnan(x_max):
            high_arr[1] = dbl_min(high_arr[1], x_max)

        # handle x_next_min <= 2 delta u + x_i <= x_next_max
        if i < self.N:
            if isnan(x_next_min):
                self.a_arr[i, 0] = 0
                self.b_arr[i, 0] = 0
                self.c_arr[i, 0] = -1
            else:
                self.a_arr[i, 0] = - 2 * self.deltas[i]
                self.b_arr[i, 0] = - 1.0
                self.c_arr[i, 0] = x_next_min
            if isnan(x_next_max):
                self.a_arr[i, 1] = 0
                self.b_arr[i, 1] = 0
                self.c_arr[i, 1] = -1
            else:
                self.a_arr[i, 1] = 2 * self.deltas[i]
                self.b_arr[i, 1] = 1.0
                self.c_arr[i, 1] = - x_next_max
        else:
            # at the last stage, neglect this constraint
            self.a_arr[i, 0:2] = 0
            self.b_arr[i, 0:2] = 0
            self.c_arr[i, 0:2] = -1

        # objective function
        self.v[0] = - g[0]
        self.v[1] = - g[1]

        # warmstarting feature: one in two solvers, upper and lower,
        # is be selected depending on the sign of g[1]

        # solver selected: upper solver. This is selected when
        # computing the lower bound of the controllable set.
        if g[1] > 0:  
            # print "v={:}\n a={:}\n b={:}\n c={:}\n low={:}\n high={:}\n active_c_up={:}".format(
            #     *map(repr, map(np.asarray,
            #                    [self.v, self.a_arr[i], self.b_arr[i], self.c_arr[i],
            #                     low_arr, high_arr, self.active_c_up])))
            solution = cy_solve_lp2d(self.v, self.a_arr[i], self.b_arr[i], self.c_arr[i],
                                     low_arr, high_arr, self.active_c_up,
                                     True, self.index_map, self.a_1d, self.b_1d)
            if solution.result == 0:
                # print("upper solver fails")
                var[0] = NAN
                var[1] = NAN
            else:
                var[:] = solution.optvar
                self.active_c_up[0] = solution.active_c[0]
                self.active_c_up[1] = solution.active_c[1]

        # solver selected: lower solver. This is when computing the
        # lower bound of the controllable set, or computing the
        # parametrization in the forward pass
        else:
            solution = cy_solve_lp2d(self.v, self.a_arr[i], self.b_arr[i], self.c_arr[i],
                                     low_arr, high_arr, self.active_c_down,
                                     True, self.index_map, self.a_1d, self.b_1d)
            if solution.result == 0:
                # print("lower solver fails")
                # print "v={:}\n a={:}\n b={:}\n c={:}\n low={:}\n high={:}".format(
                    # *map(repr, map(np.asarray,
                                  # [self.v, self.a_arr[i], self.b_arr[i], self.c_arr[i], self.low_arr[i], self.high_arr[i]])))
                var[0] = NAN
                var[1] = NAN
            else:
                var[:] = solution.optvar
                self.active_c_down[0] = solution.active_c[0]
                self.active_c_down[1] = solution.active_c[1]
                # print "v={:}\n a={:}\n b={:}\n c={:}\n low={:}\n high={:}\n result={:}\n-----".format(
                    # *map(repr, map(np.asarray,
                                   # [self.v, self.a_arr[i], self.b_arr[i], self.c_arr[i], self.low_arr[i], self.high_arr[i], var])))
                # print np.asarray(self.active_c_down)
                
        return var

    def setup_solver(self):
        pass

    def close_solver(self):
        pass

