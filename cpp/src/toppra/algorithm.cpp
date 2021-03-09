#include <toppra/algorithm.hpp>

#include <algorithm>
#include <cstddef>
#include <iostream>
#include "toppra/toppra.hpp"

#include <ctime>
#include <sys/time.h>

unsigned long long GetCurrentTimeUsec()
{
  struct timeval tv;
  gettimeofday(&tv,NULL);
  return ((unsigned long long)tv.tv_sec * 1000000 + (unsigned long long)tv.tv_usec);
}

namespace toppra {

PathParametrizationAlgorithm::PathParametrizationAlgorithm(
    LinearConstraintPtrs constraints, const GeometricPathPtr &path)
    : m_constraints(std::move(constraints)), m_path(path) {};

ReturnCode PathParametrizationAlgorithm::computePathParametrization(value_type vel_start,
                                                                    value_type vel_end) {
  size_t be = GetCurrentTimeUsec();
  initialize();
  size_t en = GetCurrentTimeUsec();
  printf("time of init:                             %fms\n", (en - be)/1000.0);
  be = en;
  m_solver->setupSolver();
  Bound vel_ends;
  vel_ends.setConstant(vel_end);
  m_data.ret_code = computeControllableSets(vel_ends);
  en = GetCurrentTimeUsec();
  printf("time of computeControllableSets:          %fms\n", (en - be)/1000.0);
  be = en;
  if ((int)m_data.ret_code > 0) {
    return m_data.ret_code;
  }
  m_data.ret_code = computeForwardPass(vel_start);
  en = GetCurrentTimeUsec();
  printf("time of computeForwardPass:               %fms\n", (en - be)/1000.0);
  return m_data.ret_code;
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
  Bound x (m_initXBound), x_next;
  for (std::size_t i = m_N - 1; i != (std::size_t)-1; i--) {
    TOPPRA_LOG_DEBUG(i << ", " << m_N);
    x_next = m_data.controllable_sets.row(i + 1);
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

ReturnCode PathParametrizationAlgorithm::computePathParametrizationParallel(value_type vel_start,
                                                                            value_type vel_end) {

  size_t be = GetCurrentTimeUsec();
  initialize();
  size_t en = GetCurrentTimeUsec();
  printf("time of init:                             %fms\n", (en - be)/1000.0);
  be = en;

  m_solver->setupSolver();
  Bound vel_ends;
  vel_ends.setConstant(vel_end);
  m_data.ret_code = computeControllableSetsParallel(vel_ends);
  en = GetCurrentTimeUsec();
  printf("time of computeControllableSetsParallel:  %fms\n", (en - be)/1000.0);
  be = en;
  if ((int)m_data.ret_code > 0) {
    return m_data.ret_code;
  }

  m_data.ret_code = computeForwardPass(vel_start);
  en = GetCurrentTimeUsec();
  printf("time of computeForwardPass:               %fms\n", (en - be)/1000.0);
  return m_data.ret_code;
};

ReturnCode PathParametrizationAlgorithm::computeControllableSetsParallel(
    const Bound &vel_ends) {
  ReturnCode ret = ReturnCode::OK;
  bool solver_ret;
  Vector g_upper{2}, g_lower{2}, solution;
  // Vectors  solution_upper, solution_lower;
  // solution_upper.assign(m_N, Vector::Zero(2, 1));
  // solution_lower.assign(m_N, Vector::Zero(2, 1));

  g_upper << 1e-9, -1;
  g_lower << -1e-9, 1;
  m_data.controllable_sets.row(m_N) = vel_ends.cwiseAbs2();

  size_t be = GetCurrentTimeUsec();

  bool batch_ok = true;
  printf("omp omp_get_max_threads %d\n", omp_get_max_threads());
  #pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < m_N; i++) {
    // solver_ret = m_solver->solveStagewiseBatch(i, g_upper, solution_upper[i]);
    solver_ret = m_solver->solveStagewiseBatch(i, g_upper);
    if (!solver_ret) {
      ret = ReturnCode::ERR_FAIL_CONTROLLABLE;
      printf("Fail: solveStagewiseBatch, upper problem, idx: %d\n", i);
      batch_ok = false;
    }

    // solver_ret = m_solver->solveStagewiseBatch(i, g_lower, solution_lower[i]);
    solver_ret = m_solver->solveStagewiseBatch(i, g_lower);
    if (!solver_ret) {
      ret = ReturnCode::ERR_FAIL_CONTROLLABLE;
      printf("Fail: solveStagewiseBatch, lower problem, idx: %d\n", i);
      batch_ok = false;
    }
  }
  if (!batch_ok) return ret;

  size_t en = GetCurrentTimeUsec();
  printf("time of Batch:    %fms\n", (en - be)/1000.0);
  be = en;

  Bound x_next;
  for (int i = m_N - 1; i != -1; i--) {
    x_next = m_data.controllable_sets.row(i + 1);
    solver_ret = m_solver->solveStagewiseBack(i, g_upper, x_next, solution);
    if (!solver_ret) {
      ret = ReturnCode::ERR_FAIL_CONTROLLABLE;
      printf("Fail: solveStagewiseBack, upper problem, idx: %d\n", i);
      break;
    }
    m_data.controllable_sets(i, 1) = solution[1];

    solver_ret = m_solver->solveStagewiseBack(i, g_lower, x_next, solution);
    if (!solver_ret) {
      ret = ReturnCode::ERR_FAIL_CONTROLLABLE;
      printf("Fail: solveStagewiseBack, lower problem, idx: %d\n", i);
      break;
    }
    // For whatever reason, sometimes the solver return negative
    // solution despite having a set of positve bounds. This readjust
    // if the solution is negative.
    m_data.controllable_sets(i, 0) = std::max(0.0, solution[1]);
  }
  en = GetCurrentTimeUsec();
  printf("time of Back:     %fms\n", (en - be)/1000.0);
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

void PathParametrizationAlgorithm::setGridpoints(const Vector& gridpoints)
{
  Bound I (m_path->pathInterval());
  auto s = gridpoints.size();
  m_N = s-1;
  if (   std::fabs(gridpoints[0] - I(0)) > TOPPRA_NEARLY_ZERO
      || std::fabs(gridpoints[gridpoints.size()-1] - I(1)) > TOPPRA_NEARLY_ZERO)
    throw std::invalid_argument("Path interval does not match first and last gridpoints.");
  if (((gridpoints.tail(s-1) - gridpoints.head(s-1)).array() <= 0).any())
    throw std::invalid_argument("gridpoints should be strictly increasing.");
  m_data.gridpoints = gridpoints;
  m_initialized = false;
}

void PathParametrizationAlgorithm::initialize() {
  if (m_initialized) return;
  if (!m_solver)
    throw std::logic_error("You must set a solver first.");
  Bound I (m_path->pathInterval());

  if (m_data.gridpoints.size() == 0)
    m_data.gridpoints = Vector::LinSpaced(m_N + 1, I(0), I(1));
  else if (m_data.gridpoints.size() != m_N + 1)
    throw std::invalid_argument("number of gridpoints does not match attribute N.");

  auto s = m_data.gridpoints.size();
  m_data.parametrization.resize(s);
  m_data.controllable_sets.resize(s, 2);
  m_data.feasible_sets.resize(s, 2);
  m_solver->initialize(m_constraints, m_path, m_data.gridpoints);
  m_initialized = true;
}

}  // namespace toppra
