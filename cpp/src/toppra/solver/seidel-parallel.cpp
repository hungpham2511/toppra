#include <toppra/solver/seidel-parallel.hpp>
#include <toppra/solver/seidel-internal.hpp>

namespace toppra {
namespace solver {

namespace seidel {
const LpSol INFEASIBLE { false };

/// Compute value coeffs * [v 1] of a constraint.
/// handling infinite values for v as, when v[i] is +/- infinity,
/// c[i]*v[i] == 0 if v[i] == 0.
template<class Coeffs, class Vars>
typename Coeffs::Scalar value(const Eigen::MatrixBase<Coeffs>& coeffs,
    const Eigen::MatrixBase<Vars>& vars)
{
  static_assert(Coeffs::RowsAtCompileTime == 1
      && Vars::ColsAtCompileTime == 1
      && Coeffs::ColsAtCompileTime == Vars::RowsAtCompileTime + 1,
      "Size mismatch between coefficient (1x(N+1)) and vars (Nx1)");
  typename Coeffs::Scalar res = coeffs(coeffs.cols()-1);
  for (int i = 0; i < Vars::RowsAtCompileTime; ++i)
    res += (coeffs[i]==0 && !std::isfinite(vars[i]) ? 0 : coeffs[i]*vars[i]);
  return res;
}

constexpr value_type infi = 1e6;

namespace internal {
  // projective coefficients to the line
  // add respective coefficients to A_1d
  template<typename Derived, typename Derived2>
  inline void project_linear_constraint (const Eigen::MatrixBase<Derived>& Aj,
      const Vector2& d_tan, const Vector2& zero_prj,
      const Eigen::MatrixBase<Derived2>& Aj_1d_)
  {
    Derived2& Aj_1d = const_cast<Derived2&>(Aj_1d_.derived());
    Aj_1d <<
      Aj.template head<2>() * d_tan,
      value(Aj, zero_prj);
  }
}

}


void SeidelParallel::initialize (const LinearConstraintPtrs& constraints, const GeometricPathPtr& path,
        const Vector& times)
{
  Solver::initialize (constraints, path, times);

  // Currently only support Canonical Linear Constraint
  assert(nbVars() == 2);

  // structural variables
  nC = 2;
  for (const Solver::LinearConstraintParams& linParam : m_constraintsParams.lin)
    nC += linParam.F[0].rows();

  N = nbStages();

  // init constraint coefficients for the 2d lps, which are
  // m_A, m_low, m_high.
  // The first dimensions of these array all equal N + 1.
  m_A.assign(N+1, MatrixX3::Zero(nC, 3));
  m_low  = MatrixX2::Constant(N + 1, 2, -seidel::infi);
  m_high = MatrixX2::Constant(N + 1, 2,  seidel::infi);
  int cur_index = 0;

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

  for (int i = 0; i < N+1; ++i) {
    m_low (i, 1) = std::max(m_low (i, 1), 0.0);
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
  m_A_1d.assign(N+1, MatrixX2::Zero(nC + 4, 2));

  // m_solution.assign(N+1, seidel::LpSol());
  m_solution_upper.assign(N+1, seidel::LpSol());
  m_solution_lower.assign(N+1, seidel::LpSol());

}



#define TOPPRA_SEIDEL_LP2D(w,X)                         \
  TOPPRA_LOG_##w("Seidel LP 2D:\n"                      \
      << "v: " << v << '\n'                             \
      << "A:\n" << A << '\n'                            \
      << low[0] << "\t<= x[0] <= " << high[0] << '\n'    \
      << low[1] << "\t<= x[1] <= " << high[1] << '\n'    \
      << X);

void SeidelParallel::solve_lp2d_parallel(bool isback, const RowVector2 v, const MatrixX3& A,
    const Vector2& low, const Vector2& high, MatrixX2& A_1d, seidel::LpSol& sol)
{
  assert(A_1d.rows() == A.rows()+4);
  sol.feasible = false;

  // number of working set recomputation
  unsigned int nrows = isback ? A.rows() : A.rows() - 2;
  Vector2 cur_optvar = sol.optvar;

  // absolute bounds used in solving the 1 dimensional
  // optimization sub-problems. These bounds needs to be very
  // large, so that they are never active at the optimization
  // solution of these 1D subproblem..

  // print all input to the algorithm
  TOPPRA_SEIDEL_LP2D(DEBUG, "");

  // handle fixed bounds (low, high). The following convention is
  // adhered to: fixed bounds are assigned the numbers: -1, -2, -3,
  // -4 according to the following order: low[0], high[0], low[1],
  // high[1].
  // for (int i = 0; i < 2; ++i) {
  for (int i = 0; !isback && i < 2; ++i) {
    if (low[i] > high[i]) {
      // If the difference between low and high is sufficiently small, then
      // return infeasible.
      if (   low[i] - high[i] > std::max(std::abs(low[i]),std::abs(high[i]))* seidel::REL_TOLERANCE
          || low[i] == seidel::infi
          || high[i] == -seidel::infi) {
        TOPPRA_SEIDEL_LP2D(WARN,
            "-> incoherent bounds. high - low = " << (high - low).transpose());
        return;
      }
      // Otherwise, assume variable i is static. Thus we are left with a 1D LP.
      int j = 1-i;
      sol.optvar[i] = (low[i]+high[i])/2;
      A_1d.topRows(nrows) << A.col(j), A.col(2) + sol.optvar[i] * A.col(i);
      A_1d.middleRows<2>(nrows) << -1,   low[j],
                                    1, -high[j];

      seidel::LpSol1d sol_1d = seidel::solve_lp1d({ v[j], 0. }, A_1d.topRows(nrows+2));
      if (!sol_1d.feasible) {
        TOPPRA_SEIDEL_LP2D(WARN, "-> infeasible 1");
        return;
      }
      sol.optvar[j] = sol_1d.optvar;
      sol.feasible = true;
      sol.optval = v * sol.optvar;
    }

    cur_optvar[i]   = v[i] > 0 ? high[i] : low[i];
  }
  TOPPRA_LOG_DEBUG("cur_optvar = " << cur_optvar.transpose());

  // pre-process the inequalities, remove those that are redundant

  // handle other constraints in a, b, c
  int ib;
  if (!isback) {
    ib = 0;
  } else {
    // ib = 0;
    ib = nrows - 2;
  }
  for (int i = ib; i < nrows; ++i) {
    // if current optimal variable satisfies the i-th constraint, continue
    if (seidel::value(A.row(i), cur_optvar) < seidel::ABS_TOLERANCE)
      continue;
    // otherwise, project all constraints on the line defined by (a[i], b[i], c[i])
    // project the origin (0, 0) onto the new constraint
    // let ax + by + c=0 b the new constraint
    // let zero_prj be the projected point, one has
    //     zero_prj =  1 / (a^2 + b^2) [a  -b] [-c]
    //                                 [b   a] [ 0]
    // this can be derived using perpendicularity
    Vector2 zero_prj (A.row(i).head<2>());
    zero_prj *= - A(i,2) / zero_prj.squaredNorm();

    // Let x = zero_prj + d_tan * t
    // where t is the variable of the future 1D LP.
    Vector2 d_tan { -A(i,1), A(i,0) }; // vector parallel to the line
    Vector2 v_1d { v * d_tan, 0 };

    // Size of the 1D LP: 4 + k
    // Compute the constraint parameters of the 1D LP corresponding to the
    // linear constraints.
    for (int j = 0; j < i; ++j)
      seidel::internal::project_linear_constraint(A.row(j),
              d_tan, zero_prj, A_1d.row(j));
    //A_1d.topRows(k) << A.topRows(k) * d_tan, value(A.topRows(k), zero_prj);
    // Compute the constraint parameters of the 1D LP corresponding to the
    // 4 bound constraints.
    // handle low <= x
    seidel::internal::project_linear_constraint(RowVector3 { -1., 0., low[0] },
        d_tan, zero_prj, A_1d.row(i));
    // handle x <= high
    seidel::internal::project_linear_constraint(RowVector3 { 1., 0., -high[0] },
        d_tan, zero_prj, A_1d.row(i+1));
    // handle low <= y
    seidel::internal::project_linear_constraint(RowVector3 { 0., -1., low[1] },
        d_tan, zero_prj, A_1d.row(i+2));
    // handle y <= high
    seidel::internal::project_linear_constraint(RowVector3 { 0., 1., -high[1] },
        d_tan, zero_prj, A_1d.row(i+3));

    // solve the projected, 1 dimensional LP
    TOPPRA_LOG_DEBUG("Seidel LP 2D activate constraint " << i);
    seidel::LpSol1d sol_1d = seidel::solve_lp1d(v_1d, A_1d.topRows(4+i));
    TOPPRA_LOG_DEBUG("Seidel LP 1D solution:\n" << sol_1d);

    if (!sol_1d.feasible) {
      TOPPRA_SEIDEL_LP2D(WARN, "-> infeasible 2");
      return;
    }

    // feasible, wrapping up
    // compute the current optimal solution
    cur_optvar = zero_prj + sol_1d.optvar * d_tan;
    TOPPRA_LOG_DEBUG("i = " << i << ". cur_optvar = " << cur_optvar.transpose());
  }

  // Sanity check for debugging purpose.
  for (int i = 0; i < nrows; ++i) {
    value_type v = seidel::value(A.row(i), cur_optvar);
    if (v > 0.0) {
      TOPPRA_LOG_DEBUG("Seidel 2D: contraint " << i << " violated (" << v << " should be <= 0).");
    }
  }

  TOPPRA_LOG_DEBUG("Seidel solution: " << cur_optvar.transpose());

  // Fill the solution struct
  sol.feasible = true;
  sol.optvar = cur_optvar;
  sol.optval = v * sol.optvar;
}

#undef TOPPRA_SEIDEL_LP2D


// bool SeidelParallel::solveStagewiseBatch(int i, const Vector& g, Vector& solution)
bool SeidelParallel::solveStagewiseBatch(int i, const Vector& g)
{
  assert (i <= N && 0 <= i);

  // objective function (because seidel solver does max)
  RowVector2 v (-g.head<2>());

  // warmstarting feature: one in two solvers, upper and lower,
  // is be selected depending on the sign of g[1]
  bool upper (g[1] > 0);

  if (upper) {
    solve_lp2d_parallel(false, v, m_A[i], m_low.row(i), m_high.row(i), m_A_1d[i], m_solution_upper[i]);
    if (m_solution_upper[i].feasible) {
      // solution = m_solution_upper[i].optvar;
      return true;
    }
  } else {
    solve_lp2d_parallel(false, v, m_A[i], m_low.row(i), m_high.row(i), m_A_1d[i], m_solution_lower[i]);
    if (m_solution_lower[i].feasible) {
      // solution = m_solution_lower[i].optvar;
      return true;
    }
  }

  TOPPRA_LOG_DEBUG("Seidel: solver fails (upper ? " << upper << ')');
  // printf("Seidel: solver fails %s, i = %d\n", upper?"upper":"down", i);
  return false;
}


bool SeidelParallel::solveStagewiseBack(int i, const Vector& g, const Bound& xNext, Vector& solution)
{
  // printf("good\n");
  assert (i <= N && 0 <= i);

  // handle x_next_min <= 2 delta u + x_i <= x_next_max
  value_type delta = deltas()[i];
  if (xNext[0] <= -seidel::infi) { 
    // m_A[i].row(0) << 0, 0, -1;
    m_A[i].row(nC-2) << 0, 0, -1;
  } else {
    // m_A[i].row(0) << -2*delta, -1, xNext[0];
    m_A[i].row(nC-2) << -2*delta, -1, xNext[0];
  }
  if (xNext[1] >= seidel::infi) {
    // m_A[i].row(1) << 0, 0, -1;
    m_A[i].row(nC-1) << 0, 0, -1;
  } else {
    // m_A[i].row(1) << 2*delta, 1, -xNext[1];
    m_A[i].row(nC-1) << 2*delta, 1, -xNext[1];
  }

  // objective function (because seidel solver does max)
  RowVector2 v (-g.head<2>());

  // warmstarting feature: one in two solvers, upper and lower,
  // is be selected depending on the sign of g[1]
  bool upper (g[1] > 0);

  if (upper) {
    // m_solution_upper[i].optvar = solution;
    solve_lp2d_parallel(true, v, m_A[i], m_low.row(i), m_high.row(i), m_A_1d[i], m_solution_upper[i]);
    if (m_solution_upper[i].feasible) {
      solution = m_solution_upper[i].optvar;
      return true;
    }
  } else {
    // m_solution_lower[i].optvar = solution;
    solve_lp2d_parallel(true, v, m_A[i], m_low.row(i), m_high.row(i), m_A_1d[i], m_solution_lower[i]);
    if (m_solution_lower[i].feasible) {
      solution = m_solution_lower[i].optvar;
      return true;
    }
  }


  TOPPRA_LOG_DEBUG("Seidel: solver fails (upper ? " << upper << ')');
  // printf("Seidel: solver fails %s, i = %d\n", upper?"upper":"down", i);
  return false;
}


bool SeidelParallel::solveStagewiseOptim(std::size_t i,
        const Matrix& H, const Vector& g,
        const Bound& x, const Bound& xNext,
        Vector& solution)
{
  if (H.size() > 0)
    throw std::invalid_argument("Seidel can only solve LPs.");

  int N (nbStages());
  assert (i <= N && 0 <= i);

  // fill coefficient
  Vector2 low (m_low.row(i)),
          high (m_high.row(i));

  // handle x_min <= x_i <= x_max
  low[1] = std::max(low[1], x[0]);
  high[1] = std::min(high[1], x[1]);

  // handle x_next_min <= 2 delta u + x_i <= x_next_max
  if (i < N) {
    value_type delta = deltas()[i];
    if (xNext[0] <= -seidel::infi) // TODO isnan(x_next_min)
      m_A[i].row(0) << 0, 0, -1;
    else
      m_A[i].row(0) << -2*delta, -1, xNext[0];
    if (xNext[1] >= seidel::infi) // TODO isnan(x_next_max)
      m_A[i].row(1) << 0, 0, -1;
    else
      m_A[i].row(1) << 2*delta, 1, -xNext[1];
  } else {
    // at the last stage, neglect this constraint
    m_A[i].topLeftCorner<2,2>().setZero();
    m_A[i].topRightCorner<2,1>().setConstant(-1);
  }

  // objective function (because seidel solver does max)
  RowVector2 v (-g.head<2>());

  // warmstarting feature: one in two solvers, upper and lower,
  // is be selected depending on the sign of g[1]
  bool upper (g[1] > 0);

  seidel::LpSol lpsol = seidel::solve_lp2d(v, m_A[i], low, high, m_A_1d[i]);
  if (lpsol.feasible) {
    solution = lpsol.optvar;
    return true;
  }
  TOPPRA_LOG_DEBUG("Seidel: solver fails (upper ? " << upper << ')');
  return false;
}


} // namespace solver
} // namespace toppra
