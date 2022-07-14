#include <toppra/algorithm/toppra.hpp>

#include <memory>
#include <toppra/algorithm.hpp>
#include <toppra/toppra.hpp>

namespace toppra {
namespace algorithm {

TOPPRA::TOPPRA(LinearConstraintPtrs constraints, const GeometricPathPtr &path)
    : PathParametrizationAlgorithm{std::move(constraints), path} {
  m_solver = Solver::createDefault();
}

ReturnCode TOPPRA::computeForwardPass(value_type vel_start) {
  TOPPRA_LOG_DEBUG("computeForwardPass");
  ReturnCode ret = ReturnCode::OK;
  bool solver_ret;
  Vector g_upper{2}, solution{2};
  Matrix H;
  auto deltas = m_solver->deltas();
  Bound x, x_next;
  m_data.parametrization(0) = vel_start;
  for (std::size_t i = 0; i < m_data.parametrization.size() - 1; i++) {
    TOPPRA_LOG_DEBUG("i: " << i);
    g_upper << -2 * deltas(i), -1;
    x.setConstant(m_data.parametrization(i));
    x_next = m_data.controllable_sets.row(i + 1);
    solver_ret = m_solver->solveStagewiseOptim(i, H, g_upper, x, x_next, solution);
    if (!solver_ret) {
      ret = ReturnCode::ERR_FAIL_FORWARD_PASS;
      TOPPRA_LOG_DEBUG("Fail: forward pass, idx: " << i);
      break;
    }
    /// \todo This can be optimized further by solving a 1D problem instead of 2D
    // Claim the output to be within the controllable sets.
    m_data.parametrization(i + 1) =
        std::min(m_data.controllable_sets(i + 1, 1),
                 std::max(m_data.controllable_sets(i + 1, 0),
                          m_data.parametrization(i) + 2 * deltas(i) * solution(0)));
    TOPPRA_LOG_DEBUG("Ok: u[" << i << "]= " << solution(0) << "x[" << i + 1
                              << "]=" << m_data.parametrization(i + 1));
  }

  return ret;
};

}  // namespace algorithm
}  // namespace toppra
