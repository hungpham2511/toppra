#include <memory>
#include <toppra/algorithm.hpp>
#include <toppra/algorithm/toppra.hpp>
#include <toppra/toppra.hpp>

namespace toppra {
namespace algorithm {

TOPPRA::TOPPRA(const LinearConstraintPtrs &constraints, const GeometricPath &path)
    : PathParametrizationAlgorithm{constraints, path} {};


ReturnCode TOPPRA::computeForwardPass(double vel_start) {
  ReturnCode ret = OK;
  bool solver_ret;
  Vector g_upper{2}, solution;
  Matrix H;
  auto deltas = m_solver->deltas();
  // std::cout << deltas << std::endl;
  Bound x, xNext;
  m_parametrization(0) = vel_start;
  for (int i=0; i < m_N; i++){
    g_upper << -2 * deltas(i), -1;
    x << m_parametrization(i), m_parametrization(i);
    xNext << m_controllable_sets(i + 1, 0), m_controllable_sets(i + 1, 1);
    solver_ret = m_solver->solveStagewiseOptim(i, H, g_upper, x, xNext, solution);
    if (!solver_ret) {
      ret = ERR_FAIL_FORWARD_PASS;
      std::cout << "Fail: forward pass, idx: " << i << std::endl;
      break;
    }
    // TODO: This can be optimized further by solving a 1D problem instead of 2D
    m_parametrization(i + 1) = m_parametrization(i) + 2 * deltas(i) * solution(0);
  }

  return ret;
};

}  // namespace algorithm
}  // namespace toppra
