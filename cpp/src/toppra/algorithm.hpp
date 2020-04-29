#ifndef TOPPRA_ALGORITHM_HPP
#define TOPPRA_ALGORITHM_HPP

#include <stdexcept>
#include <toppra/constraint.hpp>
#include <toppra/geometric_path.hpp>
#include <toppra/solver.hpp>
#include <toppra/toppra.hpp>

namespace toppra {

/// Return code for Path Parametrization algorithm.
enum class ReturnCode {

  /// Success
  OK = 0,

  /// Unknown error
  ERR_UNKNOWN = 1,

  /// Fail during computing controllable sets. Problem might be infeasible.
  ERR_FAIL_CONTROLLABLE = 2,

  /// Fail during forward pass. Numerical error occured.
  ERR_FAIL_FORWARD_PASS = 3,

  /// Problem is not initialized
  ERR_UNINITIALIZED = 4,

  /// Fail to ocmpute feasible sets.
  ERR_FAIL_FEASIBLE = 5,
};

struct ParametrizationData {
  /// \brief Grid-points used for solving the discretized problem.
  /// The number of points must equal m_N + 1.
  Vector gridpoints;

  ///  Output parametrization (squared path velocity)
  Vector parametrization;

  Matrix controllable_sets;
  Matrix feasible_sets;

  ///  Return code of the algorithm.
  ReturnCode ret_code = ReturnCode::ERR_UNINITIALIZED;
};

/** \brief Base class for time parametrization algorithms.
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
  PathParametrizationAlgorithm(LinearConstraintPtrs constraints,
                               const GeometricPath &path);

  /** \brief Set the level of discretization used by the solver.
   *
   * If is zero, will attempt to detect automatically the most suitable grid.
   */
  void setN(int N) { m_N = N; };

  /** \brief Get output or result of algorithm.
   */
  ParametrizationData getParameterizationData() const { return m_data; };

  /** Compute the time parametrization of the given path.
   *
   * \param vel_start
   * \param vel_end
   * \return Return code.
   */
  virtual ReturnCode computePathParametrization(value_type vel_start = 0,
                                                value_type vel_end = 0);

  /** Compute the sets of feasible squared velocities.
   */
  ReturnCode computeFeasibleSets();

  virtual ~PathParametrizationAlgorithm() {}

 protected:
  /** \brief Select solver and gridpoints to use.
   *
   * This method implements a simple way to select gridpoints.
   */
  virtual void initialize();

  /** \brief Compute the forward pass.
   *
   * Derived class should provide a suitable forward pass function,
   * depending on the desired objective.
   */
  virtual ReturnCode computeForwardPass(value_type vel_start) = 0;

  /** Compute the sets of controllable squared path velocities.
   */
  ReturnCode computeControllableSets(const Bound &vel_ends);

  /** To be implemented in child method. */
  LinearConstraintPtrs m_constraints;
  const GeometricPath &m_path;
  SolverPtr m_solver;

  /// Struct containing algorithm output.
  ParametrizationData m_data;

  /// \brief Number of segments in the discretized problems.
  /// See m_gridpoints for more information.
  int m_N = 100;

  int m_initialized = false;
};

}  // namespace toppra

#endif
