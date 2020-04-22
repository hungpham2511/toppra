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

class PathParametrizationAlgorithm {
public:
  PathParametrizationAlgorithm(const LinearConstraintPtrs &constraints,
                               const GeometricPath &path)
      : m_constraints(constraints), m_path(path){};

  /** \brief Set the level of discretization used by the solver.
   *
   * If is zero, will attempt to detect automatically the most suitable grid.
   */
  void setN(int N) { m_N = N; };

  ParametrizationData getParameterizationData() const {
    return m_internal_data;
  };

  virtual int computePathParametrization(Vector &path_parametrization) = 0;
  virtual ~PathParametrizationAlgorithm() {}

protected:
  LinearConstraintPtrs m_constraints;
  const GeometricPath &m_path;
  SolverPtr m_solver;
  int m_N = 100;
  ParametrizationData m_internal_data;
};

} // namespace toppra

#endif
