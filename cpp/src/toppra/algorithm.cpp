#include <iostream>
#include <toppra/algorithm.hpp>
#include <toppra/solver/qpOASES-wrapper.hpp>

namespace toppra {

PathParametrizationAlgorithm::PathParametrizationAlgorithm(
    const LinearConstraintPtrs &constraints, const GeometricPath &path)
    : m_constraints(constraints), m_path(path){};

ReturnCode PathParametrizationAlgorithm::computePathParametrization(
    Vector &path_parametrization, double vel_start, double vel_end) {
  initialize();
  m_solver->setupSolver();
  ReturnCode ret;
  Matrix K;
  ret = computeControllableSets(K);
  if (ret > 0) {
    return ret;
  }

  Vector v = toppra::Vector::LinSpaced(0, 1, 10);
  path_parametrization = v;
  return OK;
};

ReturnCode PathParametrizationAlgorithm::computeControllableSets(
    Matrix &controllable_sets, Bound vel_ends) {
  ReturnCode ret = OK;
  bool solver_ret;
  Vector g_upper{2}, g_lower{2}, solution;
  g_upper << 1e-9, -1;
  g_lower << - 1e-9, 1;
  controllable_sets.resize(m_N + 1, 2);
  controllable_sets.setZero();
  controllable_sets(m_N, 0) = vel_ends(0);
  controllable_sets(m_N, 1) = vel_ends(1);
  Matrix H;
  Bound x, xNext;
  x << 0, 100;
  xNext << 0, 1;
  for (int i = m_N - 1; i >= 0; i--) {
    // std::cout << controllable_sets << std::endl << std::endl;
    xNext << controllable_sets(i + 1, 0), controllable_sets(i + 1, 1);
    solver_ret = m_solver->solveStagewiseOptim(m_N - 1, H, g_upper, x, xNext, solution);
    controllable_sets(i, 1) = solution[1];

    if (!solver_ret) {
      ret = ERR_FAIL_CONTROLLABLE;
      std::cout << "Fail: controllable, upper problem, idx: " << i << std::endl;
      break;
    }

    solver_ret = m_solver->solveStagewiseOptim(m_N - 1, H, g_lower, x, xNext, solution);
    controllable_sets(i, 0) = solution[1];

    if (!solver_ret) {
      ret = ERR_FAIL_CONTROLLABLE;
      std::cout << "Fail: controllable, lower problem, idx: " << i << std::endl;
      break;
    }
  }
  return ret;
}

void PathParametrizationAlgorithm::initialize() {
  m_gridpoints =
      Vector::LinSpaced(m_N + 1, m_path.pathInterval()(0), m_path.pathInterval()(1));
  m_solver =
      std::make_shared<solver::qpOASESWrapper>(m_constraints, m_path, m_gridpoints);
}

}  // namespace toppra
