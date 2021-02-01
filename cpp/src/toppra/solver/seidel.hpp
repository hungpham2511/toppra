#ifndef TOPPRA_SOLVER_SEIDEL_HPP
#define TOPPRA_SOLVER_SEIDEL_HPP

#include <memory.h>
#include <toppra/solver.hpp>

namespace toppra {
namespace solver {

/** Implementation of Seidel algorithm.
 *
 * */
class Seidel : public Solver {
  public:
    Seidel () = default;

    void initialize (const LinearConstraintPtrs& constraints, const GeometricPathPtr& path,
        const Vector& times);

    bool solveStagewiseOptim(std::size_t i,
        const Matrix& H, const Vector& g,
        const Bound& x, const Bound& xNext,
        Vector& solution);

  private:
    typedef Eigen::Matrix<value_type, Eigen::Dynamic, 2> MatrixX2;
    typedef Eigen::Matrix<value_type, Eigen::Dynamic, 3> MatrixX3;
    typedef std::vector<MatrixX3, Eigen::aligned_allocator<MatrixX3> > MatricesX3;

    MatricesX3 m_A;
    MatrixX2 m_low, m_high;

    MatrixX3 m_A_ordered;
    MatrixX2 m_A_1d;
    std::vector<int> m_index_map;
    std::array<int, 2> m_active_c_up, m_active_c_down;
}; // class Seidel

} // namespace solver
} // namespace toppra

#endif
