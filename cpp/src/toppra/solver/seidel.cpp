#include <toppra/solver/seidel.hpp>
#include <toppra/solver/seidel-internal.hpp>

namespace toppra {
namespace solver {

namespace seidel {
template<class Coeffs, class Vars>
typename Coeffs::Scalar value(const Eigen::MatrixBase<Coeffs>& coeffs,
    const Eigen::MatrixBase<Vars>& vars)
{
  static_assert(Coeffs::RowsAtCompileTime == 1
      && Vars::ColsAtCompileTime == 1
      && Coeffs::ColsAtCompileTime == Vars::RowsAtCompileTime + 1,
      "Size mismatch between coefficient (1x(N+1)) and vars (Nx1)");
  return coeffs.template head<Vars::RowsAtCompileTime>() * vars + coeffs[Coeffs::ColsAtCompileTime-1];
}

namespace internal {
  // projective coefficients to the line
  // add respective coefficients to A_1d
  template<typename Derived>
  void compute_denom_and_t_limit (const Eigen::MatrixBase<Derived>& Aj,
      const Vector2& d_tan, const Vector2& zero_prj,
      value_type& denom, value_type& t_limit)
  {
    denom = Aj.template head<2>() * d_tan;
    t_limit = value(Aj, zero_prj);
  }
}

LpSol solve_lp2d(const RowVector2& v,
    const MatrixX3& A,
    const Vector2& low, const Vector2& high,
    std::array<int, 2> active_c, bool use_cache,
    std::vector<int> index_map,
    MatrixX2& A_1d)
{
  // number of working set recomputation
  unsigned int nrows = A.rows();
  Vector2 cur_optvar;

  // absolute bounds used in solving the 1 dimensional
  // optimization sub-problems. These bounds needs to be very
  // large, so that they are never active at the optimization
  // solution of these 1D subproblem..
  LpSol sol;

  // print all input to the algorithm
  TOPPRA_LOG_DEBUG("Seidel LP 2D:\n"
      << "v: " << v << '\n'
      << "A:\n" << A << '\n'
      << "lo: " << low .transpose() << '\n'
      << "hi: " << high.transpose() << '\n'
      );

  if (use_cache) {
    assert(index_map.size() == nrows);
  } else {
    index_map.resize(nrows);
    for (int i = 0; i < nrows; ++i) index_map[i] = i;
    A_1d = MatrixX2::Zero(nrows+4, 2);
  }

  // handle fixed bounds (low, high). The following convention is
  // adhered to: fixed bounds are assigned the numbers: -1, -2, -3,
  // -4 according to the following order: low[0], high[0], low[1],
  // high[1].
  for (int i = 0; i < 2; ++i) {
    if (low[i] > high[i]) {
      if (low[i] > high[i] + TINY) {
        TOPPRA_LOG_WARN("Seidel LP 2D:\n"
            << "v: " << v << '\n'
            << "A:\n" << A << '\n'
            << "lo: " << low .transpose() << '\n'
            << "hi: " << high.transpose() << '\n'
            << "-> incoherent bounds. high - low = " << (high - low).transpose());
        return INFEASIBLE;
      }
      // Assume variable i is static.
      // Remove it and call lp1d
      int j = 1-i;
      sol.optvar[i] = (low[i]+high[i])/2;
      A_1d.topRows(nrows) << A.col(j), A.col(2) + sol.optvar[i] * A.col(i);
      A_1d.row(nrows  ) << -1,   low[j];
      A_1d.row(nrows+1) <<  1, -high[j];

      LpSol sol_1d = solve_lp1d({ v[j], 0. }, A_1d.topRows(nrows+2));
      if (!sol_1d.feasible) {
        TOPPRA_LOG_WARN("Seidel LP 2D:\n"
            << "v: " << v << '\n'
            << "A:\n" << A << '\n'
            << "lo: " << low .transpose() << '\n'
            << "hi: " << high.transpose() << '\n'
            << "-> infeasible");
        return INFEASIBLE;
      }
      sol.optvar[j] = sol_1d.optvar[0];
      sol.feasible = true;
      sol.optval = v * sol.optvar;
      sol.active_c[0] = -2*i - (v[i] > 0 ? 2 : 1);
      switch(sol_1d.active_c[0] - nrows) {
        case 0: sol.active_c[1] = LOW (j); break;
        case 1: sol.active_c[1] = HIGH(j); break;
        default: sol.active_c[1] = sol_1d.active_c[0];
      }
      return sol;
    }

    if (v[i] > TINY) {
      cur_optvar[i] = high[i];
      sol.active_c[i] = HIGH(i);
    } else {
      cur_optvar[i] = low[i];
      sol.active_c[i] = LOW(i);
    }
  }
  TOPPRA_LOG_DEBUG("cur_optvar = " << cur_optvar.transpose());

  // If active_c contains valid entries, swap the first two indices
  // in index_map to these values.
  unsigned int cur_row = 2;
  if (   active_c[0] >= 0 && active_c[0] < nrows
      && active_c[1] >= 0 && active_c[1] < nrows
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

  // handle other constraints in a, b, c
  for (int k = 0; k < nrows; ++k) {
    int i = index_map[k];
    // if current optimal variable satisfies the i-th constraint, continue
    if (value(A.row(i), cur_optvar) < TINY)
      continue;
    // otherwise, project all constraints on the line defined by (a[i], b[i], c[i])
    sol.active_c[0] = i;
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

    // project 4 + k constraints onto the parallel line. each
    // constraint occupies a row of A_1d.
    for (int j = 0; j < 4 + k; ++j) { // nb rows 1d.
      value_type denom, t_limit;
      switch (j-k) {
        case 0: // j == k: handle low <= x
          internal::compute_denom_and_t_limit(RowVector3 { -1., 0., low[0] },
              d_tan, zero_prj, denom, t_limit);
          break;
        case 1: // j == k + 1: handle x <= high
          internal::compute_denom_and_t_limit(RowVector3 { 1., 0., -high[0] },
              d_tan, zero_prj, denom, t_limit);
          break;
        case 2: // j == k + 2: handle low <= y
          internal::compute_denom_and_t_limit(RowVector3 { 0., -1., low[1] },
              d_tan, zero_prj, denom, t_limit);
          break;
        case 3: // j == k + 3: handle y <= high
          internal::compute_denom_and_t_limit(RowVector3 { 0., 1., -high[1] },
              d_tan, zero_prj, denom, t_limit);
          break;
        default: // handle other constraint
          internal::compute_denom_and_t_limit(A.row(index_map[j]),
              d_tan, zero_prj, denom, t_limit);
      }

      if (denom > TINY)
        A_1d.row(j) <<  1.,  t_limit / denom;
      else if (denom < -TINY)
        A_1d.row(j) << -1., -t_limit / denom;
      else {
        // Current constraint is parallel to the base one.
        // if infeasible, return failure.
        if (t_limit > SMALL) {
          TOPPRA_LOG_DEBUG("Seidel: infeasible constraint. t_limit = " << t_limit
              << ", denom = " << denom);
          return INFEASIBLE;
        }
        // Otherwise, specify 0 <= 1
        A_1d.row(j) << 0, -1.;
      }
    }

    // solve the projected, 1 dimensional LP
    TOPPRA_LOG_DEBUG("Seidel LP 2D activate constraint " << i);
    LpSol sol_1d = solve_lp1d(v_1d, A_1d.topRows(4+k));
    TOPPRA_LOG_DEBUG("Seidel LP 1D solution:\n" << sol_1d);

    // 1d lp infeasible
    if (!sol_1d.feasible) {
      TOPPRA_LOG_WARN("Seidel LP 2D:\n"
          << "v: " << v << '\n'
          << "A:\n" << A << '\n'
          << "lo: " << low .transpose() << '\n'
          << "hi: " << high.transpose() << '\n'
          << "-> infeasible");
      return INFEASIBLE;
    }

    // feasible, wrapping up
    // compute the current optimal solution
    cur_optvar = zero_prj + sol_1d.optvar[0] * d_tan;
    TOPPRA_LOG_DEBUG("cur_optvar = " << cur_optvar.transpose());
    // record the active constraint's index
    assert(sol_1d.active_c[0] >= 0 && sol_1d.active_c[k] < k+4);
    if (sol_1d.active_c[0] >= k) // Bound constraint
      sol.active_c[1] = k - sol_1d.active_c[0] - 1;
    else // Linear constraint
      sol.active_c[1] = index_map[sol_1d.active_c[0]];
  }

  for (int i = 0; i < nrows; ++i) {
    value_type v = value(A.row(i), cur_optvar);
    if (v > 0) {
      TOPPRA_LOG_DEBUG("Seidel 2D: contraint " << i << " violated (" << v << " should be <= 0).");
    }
  }

  TOPPRA_LOG_DEBUG("Seidel solution: " << cur_optvar.transpose()
      << "\n active constraints " << sol.active_c[0] << " " << sol.active_c[1]);

  // Fill the solution struct
  sol.feasible = true;
  sol.optvar = cur_optvar;
  sol.optval = v * sol.optvar;
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
  // m_A, m_low, m_high.
  // The first dimensions of these array all equal N + 1.
  m_A.assign(N+1, MatrixX3::Zero(nC, 3));
  m_low  = MatrixX2::Constant(N + 1, 2, seidel::VAR_MIN);
  m_high = MatrixX2::Constant(N + 1, 2, seidel::VAR_MAX);
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
  m_index_map.resize(nC, 0);
  m_active_c_up.fill(0);
  m_active_c_down.fill(0);
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
  Vector2 low (m_low.row(i)),
          high (m_high.row(i));

  // handle x_min <= x_i <= x_max
  low[1] = std::max(low[1], x[0]);
  high[1] = std::min(high[1], x[1]);

  // handle x_next_min <= 2 delta u + x_i <= x_next_max
  if (i < N) {
    value_type delta = deltas()[i];
    if (xNext[0] < -seidel::INF) // TODO isnan(x_next_min)
      m_A[i].row(0) << 0, 0, -1;
    else
      m_A[i].row(0) << -2*delta, -1, xNext[0];
    if (xNext[1] > seidel::INF) // TODO isnan(x_next_max)
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
  auto& active_c = (upper ? m_active_c_up : m_active_c_down);
  // solver selected:
  // - upper: when computing the lower bound of the controllable set.
  // - lower: when computing the lower bound of the controllable set,
  //          or computing the parametrization in the forward pass.
  seidel::LpSol lpsol = seidel::solve_lp2d(v, m_A[i],
      low, high, active_c, true, m_index_map, m_A_1d);
  if (lpsol.feasible) {
    solution = lpsol.optvar;
    active_c = lpsol.active_c;
    return true;
  }
  TOPPRA_LOG_DEBUG("Seidel: solver fails (upper ? " << upper << ')');
  return false;
}

} // namespace solver
} // namespace toppra
