#ifndef TOPPRA_SOLVER_QPOASES_WRAPPER_HPP
#define TOPPRA_SOLVER_QPOASES_WRAPPER_HPP

#include <memory.h>
#include <toppra/solver.hpp>

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
 *  \todo Add a solver that inherits from qpOASESWrapper and that uses the warm
 *  start capabilities of qpOASES
 *
 * */
class qpOASESWrapper : public Solver {
  public:
    qpOASESWrapper ();

    void initialize (const LinearConstraintPtrs& constraints, const GeometricPathPtr& path,
        const Vector& times);

    bool solveStagewiseOptim(std::size_t i,
        const Matrix& H, const Vector& g,
        const Bound& x, const Bound& xNext,
        Vector& solution);

    virtual ~qpOASESWrapper();

    value_type setBoundary () const
    {
      return m_boundary;
    }

    void setBoundary (const value_type& v)
    {
      m_boundary = v;
    }

    static void setDefaultBoundary (const value_type& v);

  private:
    /// qpOASES uses row-major storage order.
    typedef Eigen::Matrix<value_type, Eigen::Dynamic, Eigen::Dynamic,
            Eigen::RowMajor> RMatrix;
    RMatrix m_H, m_A;
    Vector m_lA, m_hA;
    value_type m_boundary;

    static value_type m_defaultBoundary;

    struct Impl;
    std::unique_ptr<Impl> m_impl;
}; // class qpOASESWrapper

} // namespace solver
} // namespace toppra

#endif
