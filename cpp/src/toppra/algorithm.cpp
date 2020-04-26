#include <cstddef>
#include <iostream>
#include <toppra/algorithm.hpp>
#include <toppra/solver/qpOASES-wrapper.hpp>
#include "toppra/toppra.hpp"

namespace toppra {

PathParametrizationAlgorithm::PathParametrizationAlgorithm(
    LinearConstraintPtrs constraints, const GeometricPath &path)
    : m_constraints(std::move(constraints)), m_path(path){};

ReturnCode PathParametrizationAlgorithm::computePathParametrization(double vel_start,
                                                                    double vel_end) {
  ReturnCode ret;
  initialize();
  m_solver->setupSolver();
  Bound vel_ends;
  vel_ends << vel_end, vel_end;
  ret = computeControllableSets(vel_ends);
  if ((int)ret > 0) {
    return ret;
  }
  ret = computeForwardPass(vel_start);
  return ret;
};

ReturnCode PathParametrizationAlgorithm::computeControllableSets(
    const Bound &vel_ends) {
  TOPPRA_LOG_DEBUG("computeControllableSets");
  ReturnCode ret = ReturnCode::OK;
  bool solver_ret;
  Vector g_upper{2}, g_lower{2}, solution;
  g_upper << 1e-9, -1;
  g_lower << -1e-9, 1;
  m_data.controllable_sets(m_N, 0) = pow(vel_ends(0), 2);
  m_data.controllable_sets(m_N, 1) = pow(vel_ends(1), 2);

  Matrix H;
  Bound x, x_next;
  x << 0, 100;
  x_next << 0, 1;
  for (std::size_t i = m_N - 1; i != (std::size_t)-1; i--) {
    TOPPRA_LOG_DEBUG(i << ", " << m_N);
    x_next << m_data.controllable_sets(i + 1, 0), m_data.controllable_sets(i + 1, 1);
    solver_ret = m_solver->solveStagewiseOptim(i, H, g_upper, x, x_next, solution);

    if (!solver_ret) {
      ret = ReturnCode::ERR_FAIL_CONTROLLABLE;
      TOPPRA_LOG_DEBUG("Fail: controllable, upper problem, idx: " << i);
      break;
    }

    m_data.controllable_sets(i, 1) = solution[1];

    solver_ret = m_solver->solveStagewiseOptim(i, H, g_lower, x, x_next, solution);

    TOPPRA_LOG_DEBUG("down: " << solution);

    if (!solver_ret) {
      ret = ReturnCode::ERR_FAIL_CONTROLLABLE;
      TOPPRA_LOG_DEBUG("Fail: controllable, lower problem, idx: " << i);
      break;
    }

    m_data.controllable_sets(i, 0) = solution[1];
  }
  return ret;
}

void PathParametrizationAlgorithm::initialize() {
  m_data.gridpoints =
      Vector::LinSpaced(m_N + 1, m_path.pathInterval()(0), m_path.pathInterval()(1));
  m_data.parametrization.resize(m_N + 1);
  m_data.controllable_sets.resize(m_N + 1, 2);
  m_solver = std::make_shared<solver::qpOASESWrapper>(m_constraints, m_path,
                                                      m_data.gridpoints);
}

}  // namespace toppra
