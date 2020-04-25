#include <memory>
#include <toppra/algorithm.hpp>
#include <toppra/algorithm/toppra.hpp>
#include <toppra/toppra.hpp>

namespace toppra {
namespace algorithm {

TOPPRA::TOPPRA(const LinearConstraintPtrs &constraints, const GeometricPath &path)
    : PathParametrizationAlgorithm{constraints, path} {};

ReturnCode TOPPRA::computeForwardPass(double vel_start) {
  TOPPRA_LOG_DEBUG("computeForwardPass");
  ReturnCode ret = ReturnCode::OK;
  bool solver_ret;
  Vector g_upper{2}, solution;
  Matrix H;
  auto deltas = m_solver->deltas();
  // std::cout << deltas << std::endl;
  Bound x, x_next;
  m_data.parametrization(0) = vel_start;
  for (std::size_t i = 0; i < m_N; i++) {
    g_upper << -2 * deltas(i), -1;
    x << m_data.parametrization(i), m_data.parametrization(i);
    x_next << m_data.controllable_sets(i + 1, 0), m_data.controllable_sets(i + 1, 1);
    solver_ret = m_solver->solveStagewiseOptim(i, H, g_upper, x, x_next, solution);
    if (!solver_ret) {
      ret = ReturnCode::ERR_FAIL_FORWARD_PASS;
      TOPPRA_LOG_DEBUG("Fail: forward pass, idx: " << i);
      break;
    }
    // TODO: This can be optimized further by solving a 1D problem instead of 2D
    m_data.parametrization(i + 1) =
        m_data.parametrization(i) + 2 * deltas(i) * solution(0);
  }

  return ret;
};

}  // namespace algorithm
}  // namespace toppra
