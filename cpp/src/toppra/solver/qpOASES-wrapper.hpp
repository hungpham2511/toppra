#ifndef TOPPRA_SOLVER_QPOASES_WRAPPER_HPP
#define TOPPRA_SOLVER_QPOASES_WRAPPER_HPP

#include <toppra/solver.hpp>

// Forward declare qpOASES solver.
namespace qpOASES {
  class SQProblem;
} // namespace qpOASES

namespace toppra {
namespace solver {

/** Wrapper around qpOASES::SQProblem
 *
 *  Internally, the problem is formulated as
 *  \f{eqnarray}
 *  min   & 0.5 y^T H y + g^T y \\
 *  s.t   & lA <= Ay <= hA      \\
 *        & l  <=  y <= h       \\
 *  \f}
 *
 * */
class qpOASESWrapper : public Solver {
  public:
    qpOASESWrapper (const LinearConstraintPtrs& constraints, const GeometricPath& path,
        const Vector& times);

    bool solveStagewiseOptim(std::size_t i,
        const Matrix& H, const Vector& g,
        const Bound& x, const Bound& xNext,
        Vector& solution);

    ~qpOASESWrapper();

  private:
    /// qpOASES uses row-major storage order.
    typedef Eigen::Matrix<value_type, Eigen::Dynamic, Eigen::Dynamic,
            Eigen::RowMajor> RMatrix;
    RMatrix m_H, m_A;
    Vector m_lA, m_hA;

    struct Impl;
    Impl* m_impl;
}; // class qpOASESWrapper

} // namespace solver
} // namespace toppra

#endif
