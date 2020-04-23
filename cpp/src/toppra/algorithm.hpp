#ifndef TOPPRA_ALGORITHM_HPP
#define TOPPRA_ALGORITHM_HPP

#include <toppra/constraint.hpp>
#include <toppra/geometric_path.hpp>
#include <toppra/solver.hpp>
#include <toppra/toppra.hpp>

namespace toppra {

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

  ParametrizationData getParameterizationData() const {
    return m_internal_data;
  };

  /** Compute the time parametrization of the given path.
   *
   * \param path_parametrization[out] The result path parametrization.
   * \param vel_start
   * \param vel_end
   * \return Return code.
   */
  virtual int computePathParametrization(Vector &path_parametrization,
                                         double vel_start = 0,
                                         double vel_end = 0);
  virtual ~PathParametrizationAlgorithm() {}

  int computeFeasibleSets(Matrix &feasible_sets);
  int computeControllableSets(Matrix &controllable_sets,
                              Bound vel_ends = Bound{0, 0});

protected:
  /** To be implemented in child method. */
  int forwardStep(int i, Bound L_current, Bound K_next, Vector & solution);
  LinearConstraintPtrs m_constraints;
  const GeometricPath &m_path;
  SolverPtr m_solver;
  int m_N = 100;
  ParametrizationData m_internal_data;
};

} // namespace toppra

#endif
