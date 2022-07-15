#ifndef TOPPRA_SOLVER_SEIDEL_PARALLEL_FINE_HPP
#define TOPPRA_SOLVER_SEIDEL_PARALLEL_FINE_HPP

#include <toppra/solver.hpp>

namespace toppra {
namespace solver {

typedef Eigen::Matrix<value_type, Eigen::Dynamic, 2> MatrixX2;
typedef Eigen::Matrix<value_type, Eigen::Dynamic, 3> MatrixX3;
typedef std::vector<MatrixX3, Eigen::aligned_allocator<MatrixX3> > MatricesX3;
typedef Eigen::Matrix<value_type, 2, 1> Vector2;
typedef Eigen::Matrix<value_type, 1, 2> RowVector2;
typedef Eigen::Matrix<value_type, 3, 1> Vector3;
typedef Eigen::Matrix<value_type, 1, 3> RowVector3;


struct LpSol1d {
  bool feasible;
  value_type optvar;
};
struct LpSol {
  bool feasible;
  value_type optval;
  Vector2 optvar;
};


const LpSol INFEASIBLE { false };

inline std::ostream& operator<< (std::ostream& os, const LpSol1d& sol)
{
  if (!sol.feasible) return os << "infeasible";
  return os << "feasible. x* = " << sol.optvar;
}
inline std::ostream& operator<< (std::ostream& os, const LpSol& sol)
{
  if (!sol.feasible) return os << "infeasible";
  return os << "feasible: " << sol.optval
    << "\n\toptvar: " << sol.optvar.transpose();
}

const LpSol1d INFEASIBLE_1D { false };


const int LOW_0  = -1,
              HIGH_0 = -2,
              LOW_1  = -3,
              HIGH_1 = -4;

const int LOW (int i) { return (i == 0) ? LOW_0  : LOW_1 ; }
const int HIGH(int i) { return (i == 0) ? HIGH_0 : HIGH_1; }

/// Avoid division by number less TINY.
const double TINY = 1e-10;
/// Authorized constraint violation threshold.
const double REL_TOLERANCE = 1e-10;
const double ABS_TOLERANCE = 1e-13;

const value_type infinity = std::numeric_limits<value_type>::infinity();



/** Implementation of SeidelParallelFine algorithm.
 *
 * */
class SeidelParallelFine : public Solver {
  public:
    SeidelParallelFine () = default;

    void initialize (const LinearConstraintPtrs& constraints, const GeometricPathPtr& path,
        const Vector& times);

    bool solveStagewiseOptim(std::size_t i,
        const Matrix& H, const Vector& g,
        const Bound& x, const Bound& xNext,
        Vector& solution);

  private:

    MatricesX3 m_A;
    MatrixX2 m_low, m_high;

    MatrixX3 m_A_ordered;
    MatrixX2 m_A_1d;
    std::vector<int> m_index_map;
    std::array<int, 2> m_active_c_up, m_active_c_down;


    template<typename Derived>
    LpSol1d solve_lp1d(const RowVector2& v, const Eigen::MatrixBase<Derived>& A);

    LpSol solve_lp2d(const RowVector2& v, const MatrixX3& A,
        const Vector2& low, const Vector2& high,
        MatrixX2& A_1d);

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

    // projective coefficients to the line
    // add respective coefficients to A_1d
    template<typename Derived, typename Derived2>
    void project_linear_constraint (const Eigen::MatrixBase<Derived>& Aj,
        const Vector2& d_tan, const Vector2& zero_prj,
        const Eigen::MatrixBase<Derived2>& Aj_1d_)
    {
      Derived2& Aj_1d = const_cast<Derived2&>(Aj_1d_.derived());
      Aj_1d <<
        Aj.template head<2>() * d_tan,
        value(Aj, zero_prj);
    }

}; // class SeidelParallelFine

} // namespace solver
} // namespace toppra

#endif
