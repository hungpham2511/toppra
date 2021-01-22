#ifndef TOPPRA_SOLVER_SEIDEL_INTERNAL_HPP
#define TOPPRA_SOLVER_SEIDEL_INTERNAL_HPP

#include <toppra/solver/seidel.hpp>

namespace toppra {
namespace solver {
typedef Eigen::Matrix<value_type, 2, 1> Vector2;
typedef Eigen::Matrix<value_type, 1, 2> RowVector2;
typedef Eigen::Matrix<value_type, 3, 1> Vector3;
typedef Eigen::Matrix<value_type, 1, 3> RowVector3;
typedef Eigen::Matrix<value_type, Eigen::Dynamic, 2> MatrixX2;
typedef Eigen::Matrix<value_type, Eigen::Dynamic, 3> MatrixX3;

namespace seidel {
constexpr int LOW_0  = -1,
              HIGH_0 = -2,
              LOW_1  = -3,
              HIGH_1 = -4;

constexpr int LOW (int i) { return (i == 0) ? LOW_0  : LOW_1 ; }
constexpr int HIGH(int i) { return (i == 0) ? HIGH_0 : HIGH_1; }

struct LpSol1d {
  bool feasible;
  value_type optvar;
  int active_c;
};
struct LpSol {
  bool feasible;
  value_type optval;
  Vector2 optvar;
  std::array<int, 2> active_c;
};

inline std::ostream& operator<< (std::ostream& os, const LpSol1d& sol)
{
  if (!sol.feasible) return os << "infeasible";
  return os << "feasible. (x* = " << sol.optvar
    << ", active_c = " << sol.active_c;
}
inline std::ostream& operator<< (std::ostream& os, const LpSol& sol)
{
  if (!sol.feasible) return os << "infeasible";
  return os << "feasible: " << sol.optval
    << "\n\toptvar: " << sol.optvar.transpose()
    << "\n\tactive_c: " << sol.active_c[0] << ' ' << sol.active_c[1]
    ;
}

constexpr LpSol1d INFEASIBLE_1D { false };

/// Avoid division by number less TINY.
constexpr double TINY = 1e-10;
/// Authorized constraint violation threshold.
constexpr double THR_VIOLATION = 1e-8;

constexpr value_type infinity = std::numeric_limits<value_type>::infinity();

#define TOPPRA_SEIDEL_LP1D(w,X)                         \
  TOPPRA_LOG_##w("Seidel LP 1D:\n"                      \
      << "v: " << v << '\n'                             \
      << "A:\n" << A << '\n'                            \
      << X);

/**
Solve a Linear Program with 1 variable.

max   v[0] x + v[1]
s.t.  A [ x 1 ] <= 0

\tparam Unitary if True, the first column of A contains only -1, 0 or 1.
**/
template<bool Unitary, typename Derived>
LpSol1d solve_lp1d_tpl(const RowVector2& v, const Eigen::MatrixBase<Derived>& A)
{
  value_type cur_min = -infinity,
             cur_max = infinity;
  int active_c_min = -1,
      active_c_max = -2;
  auto a (A.col(0)), b (A.col(1));

  TOPPRA_SEIDEL_LP1D(DEBUG, "");
  bool maximize { v[0] > 0 };

  auto ap = [&a](int i) { return (Unitary? 1.:a[i]); }; // a+
  auto an = [&a](int i) { return (Unitary?-1.:a[i]); }; // a-

  for (int i = 0; i < A.rows(); ++i) {
    if (Unitary) assert(a[i] == 0. || a[i] == 1. || a[i] == -1.);

    if (!Unitary && a[i] == 0 && b[i] > THR_VIOLATION) {
      TOPPRA_SEIDEL_LP1D(WARN, "-> constraint " << i << " infeasible.");
      return INFEASIBLE_1D;
    }
    if (a[i] > 0) {
      if (ap(i) * cur_max + b[i] > 0) { // Constraint bounds x from above
        cur_max = std::min(-b[i]/ap(i), cur_max);
        active_c_max = i;
      }
    } else if (a[i] < 0 && an(i) * cur_min + b[i] > 0) { // Constraint bounds x from below
      cur_min = std::max(-b[i]/an(i), cur_min);
      active_c_min = i;
    }
  }

  if (   cur_min - cur_max > THR_VIOLATION
      || cur_min == infinity
      || cur_max == -infinity) {
    TOPPRA_SEIDEL_LP1D(WARN, "-> incoherent bounds. high - low = "
        << cur_max << " - " << cur_min << " = " << cur_max - cur_min);
    return INFEASIBLE_1D;
  }

  if (v[0] == 0) {
    value_type x;
    if (cur_min != -infinity)
      if (cur_max != infinity) return { true, (cur_max/2+cur_min/2), active_c_min };
      else return { true, cur_min, active_c_min };
    else return { true, cur_max, active_c_max };
  }
  if (maximize)
    return LpSol1d{ true, cur_max, active_c_max };
  else
    // optimizing direction is perpencicular to the line, or is
    // pointing toward the negative direction.
    return LpSol1d{ true, cur_min, active_c_min };
}

template<typename Derived>
inline LpSol1d solve_lp1d(const RowVector2& v, const Eigen::MatrixBase<Derived>& A)
{
  return solve_lp1d_tpl<false, Derived> (v, A);
}
template<typename Derived>
inline LpSol1d solve_lp1d_atomic(const RowVector2& v, const Eigen::MatrixBase<Derived>& A)
{
  return solve_lp1d_tpl<true, Derived> (v, A);
}

#undef TOPPRA_SEIDEL_LP1D

/**
Solve a LP with two variables.

The LP is specified as follow:
        max     v^T x
        s.t.    A [ x^T 1 ]^T <= 0
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
\param A_1d A view to an initialized array. This array is created to avoid
    the cost of initializing a new array.

\return A LpSol instance that contains the ouput. When LpSol.feasible is false,
        other fields are not meaningful.
**/
LpSol solve_lp2d(const RowVector2& v,
    const MatrixX3& A,
    const Vector2& low, const Vector2& high,
    std::array<int, 2> active_c, bool use_cache,
    std::vector<int>& index_map,
    MatrixX2& A_1d);

} // namespace seidel
} // namespace solver
} // namespace toppra

#endif