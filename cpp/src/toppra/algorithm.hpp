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
                               const GeometricPathPtr &path);

  /** \brief Set the level of discretization used by the solver.
   *
   * If is zero, will attempt to detect automatically the most suitable grid.
   */
  void setN(int N) { m_N = N; m_initialized = false; };

  /** \brief Set the gridpoints (points with the path intervals)
   *
   * If not set manually, then N equally distributed points is used.
   */
  void setGridpoints(const Vector& gridpoints);

  /** \brief Set the LP/QP solver
   *
   * Default to \ref solver::qpOASESWrapper
   */
  void solver(SolverPtr solver) { m_solver.swap(solver); };

  /** \brief Get output or result of algorithm.
   */
  const ParametrizationData& getParameterizationData() const { return m_data; };

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

  /** Set initial bounds on \f$ \dot{s}^2.
   * This is helpfull when the solver encounters numerical issues.
   */
  void setInitialXBounds (const Bound& xbound)
  {
    m_initXBound = xbound;
  }

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
  GeometricPathPtr m_path;
  SolverPtr m_solver;

  /// Struct containing algorithm output.
  ParametrizationData m_data;

  /// \brief Number of segments in the discretized problems.
  /// See m_gridpoints for more information.
  int m_N = 100;

  int m_initialized = false;

  /** Set initial bounds on \f$ \dot{s}^2.
   * \sa setInitialXBounds
   * \todo The hard-coded bound below avoids numerical issues in LP / QP solvers
   * when \f$ x \f$ becomes too big. This issue should be addressed in the
   * solver wrapper themselfves as numerical behaviors is proper to each
   * individual solver.
   * See https://github.com/hungpham2511/toppra/issues/156
   */
  Bound m_initXBound = {0, 100};
};

}  // namespace toppra

#endif
