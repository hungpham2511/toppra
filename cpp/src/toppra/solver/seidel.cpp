#include <toppra/solver/seidel.hpp>

namespace toppra {
namespace solver {

typedef Eigen::Matrix<value_type, 2, 1> Vector2;
typedef Eigen::Matrix<value_type, Eigen::Dynamic, 2> MatrixX2;

namespace seidel {
struct LpSol {
  bool feasible;
  value_type optval;
  Vector2 optvar;
  std::array<int, 2> active_c;
};

std::ostream& operator<< (std::ostream& os, const LpSol& sol)
{
  if (!sol.feasible) return os << "infeasible";
  os << "feasible: " << sol.optval
    << "\n\toptvar: " << sol.optvar.transpose()
    //<< "\n\tactive_c: " << sol.active_c
    ;
}

constexpr double TINY = 1e-10;
constexpr double SMALL = 1e-8;

// bounds on variable used in seidel solver wrapper. u and x at every
// stage is constrained to stay within this range.
constexpr double VAR_MIN = -1e8;
constexpr double VAR_MAX =  1e8;

// absolute largest value that a variable can have. This bound should
// be never be reached, however, in order for the code to work properly.
constexpr double INF = 1e10;

/**
Solve a Linear Program with 1 variable.

max   v[0] x + v[1]
s.t.  a x + b <= 0
      low <= x <= high
**/
LpSol solve_lp1d(const Vector2& v, const Vector& a,
    const Vector& b, value_type low, value_type high)
{
  value_type cur_min = low,
             cur_max = high,
             cur_x;
  int active_c_min = -1,
      active_c_max = -2;

  TOPPRA_LOG_DEBUG("Seidel LP 1D:\n"
      << "v: " << v.transpose() << '\n'
      << "a: " << a.transpose() << '\n'
      << "b: " << b.transpose() << '\n'
      << "bounds: " << low << ", " << high << '\n'
      );

  for (int i = 0; i < a.size(); ++i) {
    if (a[i] > TINY) {
      cur_x = - b[i] / a[i];
      if (cur_x < cur_max) {
        cur_max = cur_x;
        active_c_max = i;
      }
    } else if (a[i] < -TINY) {
      cur_x = - b[i] / a[i];
      if (cur_x > cur_min) {
        cur_min = cur_x;
        active_c_min = i;
        //std::cout << "new min " << cur_x << std::endl;
      }
    } else {
      // a[i] is approximately zero. do nothing.
      // TODO shouldn't we check that b is zero b <= 0 ? otherwise, the problem
      // is not feasible.
      //std::cout << "a[i] == 0 for i = " << i << std::endl;
    }
  }

  if (cur_min > cur_max) {
    LpSol solution;
    solution.feasible = false;
    return solution;
  }

  if (abs(v[0]) < TINY || v[0] < 0) {
    // optimizing direction is perpencicular to the line, or is
    // pointing toward the negative direction.
    return LpSol{ true,
      v[0] * cur_min + v[1],
      { cur_min, 0. }, 
      { active_c_min, false },
    };
  } else {
    return LpSol{ true,
      v[0] * cur_max + v[1],
      { cur_max, 0. },
      { active_c_max, 0 },
    };
  }
}

/**
Solve a LP with two variables.

The LP is specified as follow:
        max     v^T [x 1]
        s.t.    a x[0] + b x[1] + c <= 0
                low <= x <= high

NOTE: A possible optimization for this function is pruning linear
constraints that are clearly infeasible. This is not implemented
because in my current code, the bottleneck is not in solving
TOPP-RA but in setting up the parameters.

Parameters
----------
\param v
\param a
\param b
\param c
\param low
\param high
\param active_c Contains (2) indicies of rows in a, b, c that are likely the
    active constraints at the optimal solution.
\param use_cache: bool
\param index_map A view to a pre-allocated integer array, to map from
    [1,...,nrows] to the considered entries. This array is created
    to avoid the cost of initializing a new array.
\param a_1d A view to an initialized array. This array is created to avoid
    the cost of initializing a new array.
\param b_1d A view to an initialized array. This array is created to avoid
    the cost of initializing a new array.

\return A LpSol instance that contains the ouput. When LpSol.feasible is false,
        other fields are not meaningful.
**/
LpSol solve_lp2d(const Vector2& v,
    const Vector& a, const Vector& b, const Vector& c,
    const Vector2& low, const Vector2& high,
    std::array<int, 2> active_c, bool use_cache,
    std::vector<int> index_map,
    Vector& a_1d,
    Vector& b_1d)
{
  // number of working set recomputation
  unsigned int nrows = a.rows();
  Vector2 cur_optvar, zero_prj,
          d_tan,                // vector parallel to the line
          v_1d {0., 0.};        // optimizing direction
  value_type aj, bj, cj;

  // absolute bounds used in solving the 1 dimensional
  // optimization sub-problems. These bounds needs to be very
  // large, so that they are never active at the optimization
  // solution of these 1D subproblem..
  double low_1d = - INF;
  double high_1d = INF;
  LpSol sol, sol_1d;

  // print all input to the algorithm
  TOPPRA_LOG_DEBUG("Seidel LP 2D:\n"
      << "v: " << v.transpose() << '\n'
      << "a: " << a.transpose() << '\n'
      << "b: " << b.transpose() << '\n'
      << "c: " << c.transpose() << '\n'
      << "lo: " << low .transpose() << '\n'
      << "hi: " << high.transpose() << '\n'
      );

  if (use_cache) {
    assert(index_map.size() == nrows);
  } else {
    index_map.resize(nrows);
    for (int i = 0; i < nrows; ++i) index_map[i] = i;
    //v_1d = np.zeros(2)  # optimizing direction
    a_1d = Vector::Zero(nrows+4);
    b_1d = Vector::Zero(nrows+4);
  }

  // handle fixed bounds (low, high). The following convention is
  // adhered to: fixed bounds are assigned the numbers: -1, -2, -3,
  // -4 according to the following order: low[0], high[0], low[1],
  // high[1].
  for (int i = 0; i < 2; ++i) {
    if (low[i] > high[i]) {
      sol.feasible = false;
      return sol;
    }
    if (v[i] > TINY) {
      cur_optvar[i] = high[i];
      sol.active_c[i] = (i == 0) ? -2 : -4;
    } else {
      cur_optvar[i] = low[i];
      sol.active_c[i] = (i == 0) ? -1 : -3;
    }
  }
  TOPPRA_LOG_DEBUG("cur_optvar = " << cur_optvar.transpose());

  // If active_c contains valid entries, swap the first two indices
  // in index_map to these values.
  unsigned int cur_row = 2;
  if (   active_c[0] >= 0
      && active_c[0] < nrows
      && active_c[1] >= 0
      && active_c[1] < nrows
      && active_c[0] != active_c[1]) {
    // active_c contains valid indices
    index_map[0] = active_c[1];
    index_map[1] = active_c[0];
    for (int i = 0; i < nrows; ++i)
      if (i != active_c[0] && i != active_c[1])
        index_map[cur_row++] = i;
  } else
    for (int i = 0; i < nrows; ++i)
      index_map[i] = i;

  // pre-process the inequalities, remove those that are redundant
  // print(np.array(index_map))

  // handle other constraints in a, b, c
  for (int k = 0; k < nrows; ++k) {
    int i = index_map[k];
    // if current optimal variable satisfies the i-th constraint, continue
    if (a[i] * cur_optvar[0] + b[i] * cur_optvar[1] + c[i] < TINY)
      continue;
    // print("a[i] * cur_optvar[0] + b[i] * cur_optvar[1] + c[i] = {:f}".format(
    //     a[i] * cur_optvar[0] + b[i] * cur_optvar[1] + c[i]))
    // print k
    // otherwise, project all constraints on the line defined by (a[i], b[i], c[i])
    sol.active_c[0] = i;
    // project the origin (0, 0) onto the new constraint
    // let ax + by + c=0 b the new constraint
    // let zero_prj be the projected point, one has
    //     zero_prj =  1 / (a^2 + b^2) [a  -b] [-c]
    //                                 [b   a] [ 0]
    // this can be derived using perpendicularity
    // more specifically
    // zero_prj[0] = -ac / (a^2 + b^2), zero_prj[1] = -bc / (a^2 + b^2)
    zero_prj << a[i], b[i];
    zero_prj *= - c[i] / (a[i]*a[i] + b[i]*b[i]);

    d_tan << -b[i], a[i];
    v_1d << d_tan.dot(v), 0;

    //if (std::fabs(v_1d[0]) < TINY)
      // std::cout << "i = " << i << '\n'
      //  << "d_tan: " << d_tan.transpose() << '\n'
      //  << "v_1d = " << v_1d[0] << std::endl;
    // project 4 + k constraints onto the parallel line. each
    // constraint occupies a row on vectors a_1d, b_1d.
    for (int j = 0; j < 4 + k; ++j) { // nb rows 1d.
      switch (j-k) {
        case 0: // j == k: handle low <= x
          aj = -1;
          bj = 0;
          cj = low[0];
          break;
        case 1: // j == k + 1: handle x <= high
          aj = 1;
          bj = 0;
          cj = -high[0];
          break;
        case 2: // j == k + 2: handle low <= y 
          aj = 0;
          bj = -1;
          cj = low[1];
          break;
        case 3: // j == k + 3: handle y <= high
          aj = 0;
          bj = 1;
          cj = -high[1];
          break;
        default: // handle other constraint
          aj = a[index_map[j]];
          bj = b[index_map[j]];
          cj = c[index_map[j]];
      }

      // projective coefficients to the line
      value_type denom = d_tan[0] * aj + d_tan[1] * bj;

      // add respective coefficients to a_1d and b_1d
      value_type t_limit = cj + zero_prj[1] * bj + zero_prj[0] * aj;
      if (denom > TINY) {
        a_1d[j] = 1.0;
        b_1d[j] = t_limit / denom;
      } else if (denom < -TINY) {
        a_1d[j] = -1.0;
        b_1d[j] = -t_limit / denom;
      } else {
        // the currently considered constraint is parallel to
        // the base one. Check if they are infeasible, in which
        // case return failure immediately.
        if (t_limit > SMALL) {
          sol.feasible = false;
          return sol;
        }
        // feasible constraints, specify 0 <= 1
        a_1d[j] = 0;
        b_1d[j] = - 1.0;
      }
    }

    // solve the projected, 1 dimensional LP
    TOPPRA_LOG_DEBUG("Seidel LP 2D activate constraint " << i);
    sol_1d = solve_lp1d(v_1d, a_1d, b_1d, low_1d, high_1d);
    TOPPRA_LOG_DEBUG("Seidel LP 1D solution:\n" << sol_1d);

    // 1d lp infeasible
    if (!sol_1d.feasible) return sol_1d;
    // feasible, wrapping up

    // print "v={:}\n a={:}\n b={:}\n low={:}\n high={:}\n nrows={:}".format(
    //     *map(repr, map(np.asarray,
    //                    [v_1d, a_1d, b_1d, low_1d, high_1d, nrows_1d])))
    // compute the current optimal solution
    cur_optvar = zero_prj + sol_1d.optvar[0] * d_tan;
    TOPPRA_LOG_DEBUG("cur_optvar = " << cur_optvar.transpose());
    // record the active constraint's index
    switch (sol_1d.active_c[0] - k) {
      case 0: sol.active_c[1] = -1; break;
      case 1: sol.active_c[1] = -2; break;
      case 2: sol.active_c[1] = -3; break;
      case 3: sol.active_c[1] = -4; break;
      default:
              if (sol_1d.active_c[0] < k)
                sol.active_c[1] = index_map[sol_1d.active_c[0]];
              else {
                // the algorithm should not reach this point. If it
                // does, this means the active constraint at the
                // optimal solution is the fixed bound used in the 1
                // dimensional subproblem. This should not happen
                // though.
                sol.feasible = false;
                return sol;
              }
    }
  }

  for (int i = 0; i < nrows; ++i) {
    value_type v = a[i] * cur_optvar[0] + b[i] * cur_optvar[1] + c[i];
    if (v > 0) {
      TOPPRA_LOG_DEBUG("Seidel 2D: contraint " << i << " violated (" << v << " should be <= 0).");
    }
  }

  TOPPRA_LOG_DEBUG("Seidel solution: " << cur_optvar.transpose()
      << "\n active constraints " << sol.active_c[0] << " " << sol.active_c[1]);

  // Fill the solution struct
  sol.feasible = true;
  sol.optvar = cur_optvar;
  sol.optval = sol.optvar.dot(v);
  return sol;
}

}

void Seidel::initialize (const LinearConstraintPtrs& constraints, const GeometricPathPtr& path,
        const Vector& times)
{
  Solver::initialize (constraints, path, times);

  // Currently only support Canonical Linear Constraint
  assert(nbVars() == 2);

  // structural variables
  int nC = 2;
  for (const Solver::LinearConstraintParams& linParam : m_constraintsParams.lin)
    nC += linParam.F[0].rows();

  int N (nbStages());

  // init constraint coefficients for the 2d lps, which are
  // self.a_arr, self.b_arr, self.c_arr, self.low_arr,
  // self.high_arr. The first dimensions of these array all equal
  // N + 1.
  a_arr    = Matrix  ::Zero(N + 1, nC);
  b_arr    = Matrix  ::Zero(N + 1, nC);
  c_arr    = Matrix  ::Zero(N + 1, nC);
  low_arr  = MatrixX2::Constant(N + 1, 2, seidel::VAR_MIN);
  high_arr = MatrixX2::Constant(N + 1, 2, seidel::VAR_MAX);
  int cur_index = 2;

  //double [:, :] ta, tb, tc
  //unsigned int nC_

  for (const Solver::LinearConstraintParams& p : m_constraintsParams.lin) {
    assert(N+1 == p.a.size());

    auto nC_ = p.F[0].rows();
    if (p.F.size() == 1) {
      /* TODO Original code seems equivalent to the following code.
       * In C++, only F and g are identical across time so I don't understand
       * why the a[i], b[i] and c[i] for i > 0 are not used.
       * */
      // <- Most time consuming code, but this computation seems unavoidable
      /*
      Vector ta = p.F[0] * p.a[0];
      Vector tb = p.F[0] * p.b[0]
      Vector tc = p.F[0] * p.c[0] - p.g[0];
      // <-- End
      for (int i = 0; i < N+1; ++i) {
        a_arr.row(i).segment(cur_index, nC_) = ta.row(i);
        b_arr.row(i).segment(cur_index, nC_) = tb.row(i);
        c_arr.row(i).segment(cur_index, nC_) = tc.row(i);
      }
      //*/

      //*
      for (int i = 0; i < N+1; ++i) {
        a_arr.row(i).segment(cur_index, nC_) = p.F[0] * p.a[i];
        b_arr.row(i).segment(cur_index, nC_) = p.F[0] * p.b[i];
        c_arr.row(i).segment(cur_index, nC_) = p.F[0] * p.c[i] - p.g[0];
      }
      //*/
    } else {
      for (int i = 0; i < N+1; ++i) {
        a_arr.row(i).segment(cur_index, nC_) = p.F[i] * p.a[i];
        b_arr.row(i).segment(cur_index, nC_) = p.F[i] * p.b[i];
        c_arr.row(i).segment(cur_index, nC_) = p.F[i] * p.c[i] - p.g[i];
      }
    }
    cur_index += nC_;
  }

  for (const Solver::BoxConstraintParams& p : m_constraintsParams.box) {
    if (!p.u.empty()) {
      assert(p.u.size() == N+1);
      for (int i = 0; i < N+1; ++i) {
        low_arr (i, 0) = std::max(low_arr (i, 0), p.u[i][0]);
        high_arr(i, 0) = std::min(high_arr(i, 0), p.u[i][1]);
      }
    }
    if (!p.x.empty()) {
      assert(p.x.size() == N+1);
      for (int i = 0; i < N+1; ++i) {
        low_arr (i, 1) = std::max(low_arr (i, 1), p.x[i][0]);
        high_arr(i, 1) = std::min(high_arr(i, 1), p.x[i][1]);
      }
    }
  }

  // init constraint coefficients for the 1d LPs
  a_1d = Vector::Zero(nC + 4);
  b_1d = Vector::Zero(nC + 4);
  index_map.resize(nC, 0);
  active_c_up.fill(0);
  active_c_down.fill(0);
}

bool Seidel::solveStagewiseOptim(std::size_t i,
        const Matrix& H, const Vector& g,
        const Bound& x, const Bound& xNext,
        Vector& solution)
{
  if (H.size() > 0)
    throw std::invalid_argument("Seidel can only solve LPs.");

  int N (nbStages());
  assert (i <= N && 0 <= i);

  // fill coefficient
  unsigned int k, cur_index = 0, j, nC;  // indices
  Vector2 low (low_arr.row(i)),
          high (high_arr.row(i));
  Vector2 v {0., 0.};

  // handle x_min <= x_i <= x_max
  low[1] = std::max(low[1], x[0]);
  high[1] = std::min(high[1], x[1]);

  // handle x_next_min <= 2 delta u + x_i <= x_next_max
  if (i < N) {
    value_type delta = deltas()[i];
    if (xNext[0] < -seidel::INF) { // TODO isnan(x_next_min)
      a_arr(i, 0) = 0;
      b_arr(i, 0) = 0;
      c_arr(i, 0) = -1;
    } else {
      a_arr(i, 0) = - 2 * delta;
      b_arr(i, 0) = - 1.0;
      c_arr(i, 0) = xNext[0];
    }
    if (xNext[1] > seidel::INF) { // TODO isnan(x_next_max)
      a_arr(i, 1) = 0;
      b_arr(i, 1) = 0;
      c_arr(i, 1) = -1;
    } else {
      a_arr(i, 1) = 2 * delta;
      b_arr(i, 1) = 1.0;
      c_arr(i, 1) = - xNext[1];
    }
  } else {
    // at the last stage, neglect this constraint
    a_arr.row(i).head<2>().setZero();
    b_arr.row(i).head<2>().setZero();
    c_arr.row(i).head<2>().setConstant(-1);
  }

  // objective function (because seidel solver does max)
  v = - g.head<2>();

  // warmstarting feature: one in two solvers, upper and lower,
  // is be selected depending on the sign of g[1]
  if (g[1] > 0) {
    // solver selected: upper solver. This is selected when
    // computing the lower bound of the controllable set.
    // print "v={:}\n a={:}\n b={:}\n c={:}\n low={:}\n high={:}\n active_c_up={:}".format(
    //     *map(repr, map(np.asarray,
    //                    [self.v, self.a_arr[i], self.b_arr[i], self.c_arr[i],
    //                     low_arr, high_arr, self.active_c_up])))
    seidel::LpSol lpsol = seidel::solve_lp2d(v, a_arr.row(i), b_arr.row(i), c_arr.row(i),
        low, high, active_c_up, false, index_map, a_1d, b_1d);
    if (lpsol.feasible) {
      solution = lpsol.optvar;
      active_c_up = lpsol.active_c;
      return true;
    }
    // print("upper solver fails")
  } else {
    // solver selected: lower solver. This is when computing the
    // lower bound of the controllable set, or computing the
    // parametrization in the forward pass
    seidel::LpSol lpsol = seidel::solve_lp2d(v, a_arr.row(i), b_arr.row(i), c_arr.row(i),
        low, high, active_c_down, false, index_map, a_1d, b_1d);
    if (lpsol.feasible) {
      solution = lpsol.optvar;
      active_c_down = lpsol.active_c;
      // print "v={:}\n a={:}\n b={:}\n c={:}\n low={:}\n high={:}\n result={:}\n-----".format(
      // *map(repr, map(np.asarray,
      // [self.v, self.a_arr[i], self.b_arr[i], self.c_arr[i], self.low_arr[i], self.high_arr[i], var])))
      // print np.asarray(self.active_c_down)
      return true;
    }
    // print("lower solver fails")
    // print "v={:}\n a={:}\n b={:}\n c={:}\n low={:}\n high={:}".format(
    // *map(repr, map(np.asarray,
    // [self.v, self.a_arr[i], self.b_arr[i], self.c_arr[i], self.low_arr[i], self.high_arr[i]])))
  }
  return false;
}

} // namespace solver
} // namespace toppra
