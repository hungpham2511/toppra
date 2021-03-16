#ifndef TOPPRA_SOLVER_GLPK_WRAPPER_HPP
#define TOPPRA_SOLVER_GLPK_WRAPPER_HPP

#include <memory.h>
#include <toppra/solver.hpp>

// Forward declare glpk solver.
struct glp_prob;

namespace toppra {
namespace solver {

/** Wrapper around GLPK library.
 *
 *  Internally, the problem is formulated as
 *  \f{eqnarray}
 *  min   & g^T y       \\
 *  s.t   & z = A y     \\
 *        & x_{min} <= x <= x_{max} \\
 *        & l_1 <= z <= h_1 \\
 *  \f}
 *  where
 *  \f{eqnarray}
 *  y =& \begin{pmatrix} u & x \end{pmatrix}^T     \\
 *  A =& \begin{pmatrix}
 *     2\delta &       1 \\
 *     F_i a_i & F_i b_i \\
 *     \vdots  & \vdots \\
 *     \end{pmatrix}\\
 *  \f}
 *
 * */
class GLPKWrapper : public Solver {
  public:
    GLPKWrapper () = default;

    void initialize (const LinearConstraintPtrs& constraints, const GeometricPathPtr& path,
        const Vector& times);

    bool solveStagewiseOptim(std::size_t i,
        const Matrix& H, const Vector& g,
        const Bound& x, const Bound& xNext,
        Vector& solution);

    virtual ~GLPKWrapper();

  private:
    glp_prob* m_lp = NULL;
}; // class GLPKWrapper

} // namespace solver
} // namespace toppra

#endif
