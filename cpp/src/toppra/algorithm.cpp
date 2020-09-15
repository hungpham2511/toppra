#include <toppra/algorithm.hpp>

#include <algorithm>
#include <cstddef>
#include <iostream>
#include "toppra/toppra.hpp"

namespace toppra {

PathParametrizationAlgorithm::PathParametrizationAlgorithm(
    LinearConstraintPtrs constraints, const GeometricPathPtr &path)
    : m_constraints(std::move(constraints)), m_path(path) {};

ReturnCode PathParametrizationAlgorithm::computePathParametrization(value_type vel_start,
                                                                    value_type vel_end) {
  ReturnCode ret;
  initialize();
  m_solver->setupSolver();
  Bound vel_ends;
  vel_ends.setConstant(vel_end);
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
  m_data.controllable_sets.row(m_N) = vel_ends.cwiseAbs2();

  Matrix H;
  Bound x, x_next;
  x << 0, 100;
  x_next << 0, 1;
  for (std::size_t i = m_N - 1; i != (std::size_t)-1; i--) {
    TOPPRA_LOG_DEBUG(i << ", " << m_N);
    x_next << m_data.controllable_sets.row(i + 1);
    solver_ret = m_solver->solveStagewiseOptim(i, H, g_upper, x, x_next, solution);

    if (!solver_ret) {
      ret = ReturnCode::ERR_FAIL_CONTROLLABLE;
      TOPPRA_LOG_DEBUG("Fail: controllable, upper problem, idx: " << i);
      break;
    }

    m_data.controllable_sets(i, 1) = solution[1];

    solver_ret = m_solver->solveStagewiseOptim(i, H, g_lower, x, x_next, solution);

    TOPPRA_LOG_DEBUG("down: " << solution.transpose());
    if (!solver_ret) {
      ret = ReturnCode::ERR_FAIL_CONTROLLABLE;
      TOPPRA_LOG_DEBUG("Fail: controllable, lower problem, idx: " << i);
      break;
    }

    // For whatever reason, sometimes the solver return negative
    // solution despite having a set of positve bounds. This readjust
    // if the solution is negative.
    m_data.controllable_sets(i, 0) = std::max(0.0, solution[1]);
  }
  return ret;
}

ReturnCode PathParametrizationAlgorithm::computeFeasibleSets() {
  initialize();
  ReturnCode ret = ReturnCode::OK;
  bool solver_ret;
  Vector g_upper{2}, g_lower{2}, solution;
  g_upper << 1e-9, -1;
  g_lower << -1e-9, 1;

  Matrix H;
  Bound x, x_next;
  x << 0, 100;
  x_next << 0, 100;
  for (std::size_t i = 0; i < m_N + 1; i++) {
    solver_ret = m_solver->solveStagewiseOptim(i, H, g_upper, x, x_next, solution);

    if (!solver_ret) {
      ret = ReturnCode::ERR_FAIL_FEASIBLE;
      TOPPRA_LOG_DEBUG("Fail: controllable, upper problem, idx: " << i);
      break;
    }

    m_data.feasible_sets(i, 1) = solution[1];

    solver_ret = m_solver->solveStagewiseOptim(i, H, g_lower, x, x_next, solution);

    if (!solver_ret) {
      ret = ReturnCode::ERR_FAIL_FEASIBLE;
      TOPPRA_LOG_DEBUG("Fail: controllable, lower problem, idx: " << i);
      break;
    }

    m_data.feasible_sets(i, 0) = solution[1];
  }
  return ret;
}

void PathParametrizationAlgorithm::initialize() {
  if (m_initialized) return;
  if (!m_solver)
    throw std::logic_error("You must set a solver first.");
  Bound I (m_path->pathInterval());
  m_data.gridpoints = Vector::LinSpaced(m_N + 1, I(0), I(1));
  m_data.parametrization.resize(m_N + 1);
  m_data.controllable_sets.resize(m_N + 1, 2);
  m_data.feasible_sets.resize(m_N + 1, 2);
  m_solver->initialize(m_constraints, m_path, m_data.gridpoints);
  m_initialized = true;
}

}  // namespace toppra
