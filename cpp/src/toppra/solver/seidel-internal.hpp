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

struct LpSol {
  bool feasible;
  value_type optval;
  Vector2 optvar;
  std::array<int, 2> active_c;
};

inline std::ostream& operator<< (std::ostream& os, const LpSol& sol)
{
  if (!sol.feasible) return os << "infeasible";
  os << "feasible: " << sol.optval
    << "\n\toptvar: " << sol.optvar.transpose()
    << "\n\tactive_c: " << sol.active_c[0] << ' ' << sol.active_c[1]
    ;
}

const LpSol INFEASIBLE { false };

constexpr double TINY = 1e-10;
constexpr double SMALL = 1e-8;

// bounds on variable used in seidel solver wrapper. u and x at every
// stage is constrained to stay within this range.
constexpr double VAR_MIN = -1e8;
constexpr double VAR_MAX =  1e8;

// absolute largest value that a variable can have. This bound should
// be never be reached, however, in order for the code to work properly.
constexpr double INF = 1e10;

constexpr value_type infinity = std::numeric_limits<value_type>::infinity();

/**
Solve a Linear Program with 1 variable.

max   v[0] x + v[1]
s.t.  A [ x 1 ] <= 0
**/
template<typename Derived>
LpSol solve_lp1d(const RowVector2& v, const Eigen::MatrixBase<Derived>& A)
{
  value_type cur_min = -infinity,
             cur_max = infinity;
  int active_c_min = -1,
      active_c_max = -2;
  auto a (A.col(0)), b (A.col(1));

  TOPPRA_LOG_DEBUG("Seidel LP 1D:\n"
      << "v: " << v << '\n'
      << "A:\n" << A << '\n'
      );
  bool feasibility { abs(v[0]) < TINY };
  bool maximize { v[0] > 0 };
  bool compute_min (feasibility || !maximize);
  bool compute_max (feasibility || maximize);

  for (int i = 0; i < a.size(); ++i) {
    if (compute_max && a[i] > TINY) {
      value_type cur_x = - b[i] / a[i];
      if (cur_x < cur_max) {
        cur_max = cur_x;
        active_c_max = i;
      }
    } else if (compute_min && a[i] < -TINY) {
      value_type cur_x = - b[i] / a[i];
      if (cur_x > cur_min) {
        cur_min = cur_x;
        active_c_min = i;
      }
    } else if (abs(a[i]) < TINY && b[i] > SMALL) {
      TOPPRA_LOG_WARN("Seidel LP 1D:\n"
          << "v: " << v << '\n'
          << "A:\n" << A << '\n'
          << "-> constraint " << i << " infeasible.");
      return INFEASIBLE;
    }
    // else a[i] is approximately zero. do nothing.
  }

  if (cur_min - cur_max > SMALL) {
    TOPPRA_LOG_WARN("Seidel LP 1D:\n"
        << "v: " << v << '\n'
        << "A:\n" << A << '\n'
        << "-> incoherent bounds. high - low = "
        << cur_max << " - " << cur_min << " = " << cur_max - cur_min);
    return INFEASIBLE;
  }

  if (maximize) {
    return LpSol{ true,
      v[0] * cur_max + v[1],
      { cur_max, 0. },
      { active_c_max, 0 },
    };
  } else {
    // optimizing direction is perpencicular to the line, or is
    // pointing toward the negative direction.
    return LpSol{ true,
      v[0] * cur_min + v[1],
      { cur_min, 0. },
      { active_c_min, 0 },
    };
  }
}

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
