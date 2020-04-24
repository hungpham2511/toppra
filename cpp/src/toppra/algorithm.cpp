#include <iostream>
#include <toppra/algorithm.hpp>
#include <toppra/solver/qpOASES-wrapper.hpp>

namespace toppra {

PathParametrizationAlgorithm::PathParametrizationAlgorithm(
    const LinearConstraintPtrs &constraints, const GeometricPath &path)
    : m_constraints(constraints), m_path(path){};

ReturnCode PathParametrizationAlgorithm::computePathParametrization(
    Vector &path_parametrization, double vel_start, double vel_end) {
  ReturnCode ret;
  initialize();
  m_solver->setupSolver();
  Bound vel_ends;
  vel_ends << vel_end, vel_end;
  ret = computeControllableSets(vel_ends);
  if (ret > 0) {
    return ret;
  }
  ret = computeForwardPass(vel_start);
  path_parametrization = m_parametrization;
  return OK;
};

ReturnCode PathParametrizationAlgorithm::computeControllableSets(Bound vel_ends) {
  ReturnCode ret = OK;
  bool solver_ret;
  Vector g_upper{2}, g_lower{2}, solution;
  g_upper << 1e-9, -1;
  g_lower << - 1e-9, 1;
  m_controllable_sets(m_N, 0) = vel_ends(0);
  m_controllable_sets(m_N, 1) = vel_ends(1);

  Matrix H;
  Bound x, xNext;
  x << 0, 100;
  xNext << 0, 1;
  for (int i = m_N - 1; i >= 0; i--) {
    // xNext << controllable_sets(i + 1, 0), controllable_sets(i + 1, 1);
    solver_ret = m_solver->solveStagewiseOptim(m_N - 1, H, g_upper, x, xNext, solution);
    // std::cout << "up: " << solution << std::endl;


    if (!solver_ret) {
      ret = ERR_FAIL_CONTROLLABLE;
      std::cout << "Fail: controllable, upper problem, idx: " << i << std::endl;
      break;
    }

    m_controllable_sets(i, 1) = solution[1];

    solver_ret = m_solver->solveStagewiseOptim(m_N - 1, H, g_lower, x, xNext, solution);
    // std::cout << "down: " << solution << std::endl;

    if (!solver_ret) {
      ret = ERR_FAIL_CONTROLLABLE;
      std::cout << "Fail: controllable, lower problem, idx: " << i << std::endl;
      break;
    }

    m_controllable_sets(i, 0) = solution[1];
  }
  return ret;
}

void PathParametrizationAlgorithm::initialize() {
  m_gridpoints =
      Vector::LinSpaced(m_N + 1, m_path.pathInterval()(0), m_path.pathInterval()(1));
  m_parametrization.resize(m_N + 1);
  m_controllable_sets.resize(m_N + 1, 2);
  m_solver =
      std::make_shared<solver::qpOASESWrapper>(m_constraints, m_path, m_gridpoints);
}
}  // namespace toppra
