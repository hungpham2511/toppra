#ifndef TOPPRA_ALGORITHM_HPP
#define TOPPRA_ALGORITHM_HPP

#include <toppra/constraint.hpp>
#include <toppra/geometric_path.hpp>
#include <toppra/solver.hpp>
#include <toppra/toppra.hpp>

namespace toppra {

enum ReturnCode {
  OK = 0,
  ERR_UNKNOWN = 1,
  ERR_FAIL_CONTROLLABLE = 2,
  ERR_FAIL_FORWARD_PASS = 3
};

struct ParametrizationData {
  Vector K, X;
  Matrix Vs;
  int ret_code = -1;
};

/** Base class for time parametrization algorithms.
 *
 */
class PathParametrizationAlgorithm {
 public:
  /** Construct the problem instance.
   *
   *  \param  constraints  List of constraints.
   *  \param  path  The geometric path.
   *
   */
  PathParametrizationAlgorithm(const LinearConstraintPtrs &constraints,
                               const GeometricPath &path);

  /** \brief Set the level of discretization used by the solver.
   *
   * If is zero, will attempt to detect automatically the most suitable grid.
   */
  void setN(int N) { m_N = N; };

  /** \brief Get output or result of algorithm.
   */
  ParametrizationData getParameterizationData() const { return m_internal_data; };

  /** Compute the time parametrization of the given path.
   *
   * \param path_parametrization[out] The result path parametrization.
   * \param vel_start
   * \param vel_end
   * \return Return code.
   */
  virtual ReturnCode computePathParametrization(Vector &path_parametrization,
                                                double vel_start = 0,
                                                double vel_end = 0);
  virtual ~PathParametrizationAlgorithm() {}

 protected:
  /** \brief Select solver and gridpoints to use.
   *
   * This method implements a simple way to select gridpoints.
   */
  virtual void initialize();
  virtual ReturnCode computeForwardPass(double vel_start) = 0;

  ReturnCode computeFeasibleSets(Matrix &feasible_sets);
  ReturnCode computeControllableSets(Bound vel_ends);

  /** To be implemented in child method. */
  ReturnCode forwardStep(int i, Bound L_current, Bound K_next, Vector &solution);
  LinearConstraintPtrs m_constraints;
  const GeometricPath &m_path;
  SolverPtr m_solver;

  /// \brief Number of segments in the discretized problems.
  /// See m_gridpoints for more information.
  int m_N = 100;

  /// \brief Grid-points used for solving the discretized problem.
  /// The number of points must equal m_N + 1.
  Vector m_gridpoints, m_parametrization;
  ParametrizationData m_internal_data;
  Matrix m_controllable_sets, m_feasible_sets;
};

}  // namespace toppra

#endif
