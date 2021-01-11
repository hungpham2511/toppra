#ifndef TOPPRA_SOLVER_SEIDEL_HPP
#define TOPPRA_SOLVER_SEIDEL_HPP

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
class Seidel : public Solver {
  public:
    Seidel () {}

    void initialize (const LinearConstraintPtrs& constraints, const GeometricPathPtr& path,
        const Vector& times);

    bool solveStagewiseOptim(std::size_t i,
        const Matrix& H, const Vector& g,
        const Bound& x, const Bound& xNext,
        Vector& solution);

  private:
    typedef Eigen::Matrix<value_type, Eigen::Dynamic, 2> MatrixX2;

    Matrix a_arr, b_arr, c_arr;
    MatrixX2 low_arr, high_arr;

    Vector a_1d, b_1d;
    std::vector<int> index_map;
    std::array<int, 2> active_c_up, active_c_down;
}; // class Seidel

} // namespace solver
} // namespace toppra

#endif
