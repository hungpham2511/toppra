#include <toppra/solver/seidel-parallel-fine.hpp>
#include <omp.h>

namespace toppra {
namespace solver {

#define TOPPRA_SEIDEL_LP1D(w,X)                         \
  TOPPRA_LOG_##w("Seidel LP 1D:\n"                      \
      << "v: " << v << '\n'                             \
      << "A:\n" << A << '\n'                            \
      << X);

#define TOPPRA_SEIDEL_LP2D(w,X)                         \
  TOPPRA_LOG_##w("Seidel LP 2D:\n"                      \
      << "v: " << v << '\n'                             \
      << "A:\n" << A << '\n'                            \
      << low[0] << "\t<= x[0] <= " << high[0] << '\n'    \
      << low[1] << "\t<= x[1] <= " << high[1] << '\n'    \
      << X);

template<typename Derived>
LpSol1d SeidelParallelFine::solve_lp1d(const RowVector2& v, const Eigen::MatrixBase<Derived>& A)
{
  // initial bounds are +/- infinity.
  value_type cur_min = -infinity,
             cur_max = infinity;

  auto a (A.col(0)), b (A.col(1));

  TOPPRA_SEIDEL_LP1D(DEBUG, "");
  bool maximize { v[0] > 0 };

  for (int i = 0; i < A.rows(); ++i) {
    // If a[i] is very small, then consider the constraint as constant.
    // TODO: Shouldn't we check instead that a[i] is much smaller that b[i] ?
    // For the following problem, what solution should be returned ?
    // max   x
    // s.t.      x -   2 <= 0
    //       eps*x - eps <= 0
    // For eps small, the code below skips the second constraint and returns 2.
    if (std::abs(a[i]) < ABS_TOLERANCE) {
      if (b[i] > ABS_TOLERANCE) {
        TOPPRA_SEIDEL_LP1D(WARN, "-> constraint " << i << " infeasible.");
        return INFEASIBLE_1D;
      }
      continue;
    }
    if (a[i] > 0) {
      if (a[i] * cur_max + b[i] > 0) { // Constraint bounds x from above
        cur_max = std::min(-b[i]/a[i], cur_max);
      }
    } else if (a[i] < 0 && a[i] * cur_min + b[i] > 0) { // Constraint bounds x from below
      cur_min = std::max(-b[i]/a[i], cur_min);
    }
  }

  if (   cur_min - cur_max > std::max(std::abs(cur_min),std::abs(cur_max))*REL_TOLERANCE
      || cur_min == infinity
      || cur_max == -infinity) {
    TOPPRA_SEIDEL_LP1D(WARN, "-> incoherent bounds. high - low = "
        << cur_max << " - " << cur_min << " = " << cur_max - cur_min);
    return INFEASIBLE_1D;
  }

  if (v[0] == 0) {
    value_type x;
    if (cur_min != -infinity)
      if (cur_max != infinity) return { true, (cur_max/2+cur_min/2)};
      else return { true, cur_min};
    else return { true, cur_max};
  }
  if (maximize)
    return LpSol1d{ true, cur_max};
  else
    // optimizing direction is perpencicular to the line, or is
    // pointing toward the negative direction.
    return LpSol1d{ true, cur_min};
}


LpSol SeidelParallelFine::solve_lp2d(const RowVector2& v, const MatrixX3& A,
    const Vector2& low, const Vector2& high,
    MatrixX2& A_1d)
{
  assert(A_1d.rows() == A.rows()+4);

  // number of working set recomputation
  unsigned int nrows = A.rows();
  Vector2 cur_optvar;

  // absolute bounds used in solving the 1 dimensional
  // optimization sub-problems. These bounds needs to be very
  // large, so that they are never active at the optimization
  // solution of these 1D subproblem..
  LpSol sol;

  // print all input to the algorithm
  TOPPRA_SEIDEL_LP2D(DEBUG, "");

  // handle constraints in a, b, c
  for (int i = 0; i < nrows; ++i) {
    // if current optimal variable satisfies the i-th constraint, continue
    if (value(A.row(i), cur_optvar) < ABS_TOLERANCE)
      continue;

    // otherwise, find the new optvar (max v[0]*optvar[0]+v[1]*optvar[1], say max v[1]*optvar[1])
    double ai = A(i, 0);
    double bi = A(i, 1);
    double ci = A(i, 2);
    double minu, maxu;
    double minx, maxx;

    // handle fixed bounds (low, high). 
    if (ai * bi < 0) {
      maxx = (-ci - ai * high[0]) / bi;
      if (maxx < high[1]) {
        maxu = high[0];
      } else {
        maxu = (-ci - bi * high[1]) / ai;
        maxx = high[1];
      }
      minx = (-ci - ai * low[0]) / bi;
      if (minx > low[1]) {
        minu = low[0];
      } else {
        minu = (-ci - bi * low[1]) / ai;
        minx = low[1];
      }
    } else {
      maxx = (-ci - ai * low[0]) / bi;
      if (maxx < high[1]) {
        maxu = low[0];
      } else {
        maxu = (-ci - bi * high[1]) / ai;
        maxx = high[1];
      }
      minx = (-ci - ai * high[0]) / bi;
      if (minx > low[1]) {
        minu = high[0];
      } else {
        minu = (-ci - bi * low[1]) / ai;
        minx = low[1];
      }
    }

    for (int j = 0; j < i; j++) {
      double aj = A(j, 0);
      double bj = A(j, 1);
      double cj = A(j, 2);
      double cross = (ai * bj - aj * bi);
      double direction = ai > 0 ? cross: -cross;
      if (direction > TINY) {
        double crossu = (bi * cj - bj * ci) / cross;
        double crossx = (ci * aj - cj * ai) / cross;
        if (crossx < maxx) {
          maxu = crossu;
          maxx = crossx;
        }
        if (maxx < minx) {
          TOPPRA_SEIDEL_LP2D(WARN, "-> infeasible");
          return INFEASIBLE;
        }
      } else if (direction < -TINY) {
        double crossu = (bi * cj - bj * ci) / cross;
        double crossx = (ci * aj - cj * ai) / cross;
        if (crossx > minx) {
          minu = crossu;
          minx = crossx;
        }
        if (maxx < minx) {
          TOPPRA_SEIDEL_LP2D(WARN, "-> infeasible");
          return INFEASIBLE;
        }
      } else {
        // otherwise two line is parallel
        if (-ci / bi * bj + cj > 0) {
          TOPPRA_SEIDEL_LP2D(WARN, "-> infeasible");
          return INFEASIBLE;
        }
      }
    }

    // if (maxx < minx) {
    //   TOPPRA_SEIDEL_LP2D(WARN, "-> infeasible");
    //   return INFEASIBLE;
    // }

    if (v[1] > 0) {
      cur_optvar[0] = maxu;
      cur_optvar[1] = maxx;
    } else {
      cur_optvar[0] = minu;
      cur_optvar[1] = minx;
    }

    TOPPRA_LOG_DEBUG("i = " << i << ". cur_optvar = " << cur_optvar.transpose());
    // record the active constraint's index
  }



  TOPPRA_LOG_DEBUG("SeidelParallelFine solution: " << cur_optvar.transpose());

  // Fill the solution struct
  sol.feasible = true;
  sol.optvar = cur_optvar;
  sol.optval = v * sol.optvar;
  return sol;
}



void SeidelParallelFine::initialize (const LinearConstraintPtrs& constraints, const GeometricPathPtr& path,
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
  // m_A, m_low, m_high.
  // The first dimensions of these array all equal N + 1.
  m_A.assign(N+1, MatrixX3::Zero(nC, 3));
  m_A_ordered.resize(nC, 3);
  m_low  = MatrixX2::Constant(N + 1, 2, -infinity);
  m_high = MatrixX2::Constant(N + 1, 2,  infinity);
  int cur_index = 2;

  for (const Solver::LinearConstraintParams& p : m_constraintsParams.lin) {
    assert(N+1 == p.a.size());

    auto nC_ = p.F[0].rows();
    const bool identical = (p.F.size() == 1);
    for (int i = 0; i < N+1; ++i) {
      int k = (identical ? 0 : i);
      m_A[i].middleRows(cur_index, nC_) <<
        p.F[k] * p.a[i], p.F[k] * p.b[i], p.F[k] * p.c[i] - p.g[k];
    }
    cur_index += nC_;
  }

  for (const Solver::BoxConstraintParams& p : m_constraintsParams.box) {
    if (!p.u.empty()) {
      assert(p.u.size() == N+1);
      for (int i = 0; i < N+1; ++i) {
        m_low (i, 0) = std::max(m_low (i, 0), p.u[i][0]);
        m_high(i, 0) = std::min(m_high(i, 0), p.u[i][1]);
      }
    }
    if (!p.x.empty()) {
      assert(p.x.size() == N+1);
      for (int i = 0; i < N+1; ++i) {
        m_low (i, 1) = std::max(m_low (i, 1), p.x[i][0]);
        m_high(i, 1) = std::min(m_high(i, 1), p.x[i][1]);
      }
    }
  }

  // init constraint coefficients for the 1d LPs
  m_A_1d = MatrixX2::Zero(nC + 4, 2);
}

bool SeidelParallelFine::solveStagewiseOptim(std::size_t i,
        const Matrix& H, const Vector& g,
        const Bound& x, const Bound& xNext,
        Vector& solution)
{
  if (H.size() > 0)
    throw std::invalid_argument("SeidelParallelFine can only solve LPs.");

  int N (nbStages());
  assert (0 <= i && i < N);

  // fill coefficient
  Vector2 low (m_low.row(i)),
          high (m_high.row(i));

  // handle x_min <= x_i <= x_max
  low[1] = std::max(low[1], x[0]);
  high[1] = std::min(high[1], x[1]);

  // handle x_next_min <= 2 delta u + x_i <= x_next_max
  if (i < N) {
    value_type delta = deltas()[i];
    m_A[i].row(0) << -2*delta, -1, xNext[0];
    m_A[i].row(1) << 2*delta, 1, -xNext[1];
  }

  // objective function (because seidel solver does max)
  RowVector2 v (-g.head<2>());

  // warmstarting feature: one in two solvers, upper and lower,
  // is be selected depending on the sign of g[1]
  bool upper (g[1] < 0);

  // solver selected:
  // - upper: when computing the upper bound of the controllable set.
  // - lower: when computing the lower bound of the controllable set,
  //          or computing the parametrization in the forward pass.
  LpSol lpsol = solve_lp2d(v, m_A[i], low, high, m_A_1d);
  if (lpsol.feasible) {
    solution = lpsol.optvar;
    return true;
  }
  TOPPRA_LOG_DEBUG("SeidelParallelFine: solver fails (upper ? " << upper << ')');
  return false;
}

} // namespace solver
} // namespace toppra
