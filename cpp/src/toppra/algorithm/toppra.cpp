#include <memory>
#include <toppra/algorithm.hpp>
#include <toppra/algorithm/toppra.hpp>
#include <toppra/solver.hpp>
#include <toppra/solver/qpOASES-wrapper.hpp>
#include <toppra/toppra.hpp>

namespace toppra {
namespace algorithm {

TOPPRA::TOPPRA(const LinearConstraintPtrs &constraints,
               const GeometricPath &path)
    : PathParametrizationAlgorithm{constraints, path} {};

int TOPPRA::computePathParametrization(Vector &path_parametrization) {
  m_solver = std::make_shared<solver::qpOASESWrapper>(
      m_constraints, m_path,
      Vector::LinSpaced(m_N, m_path.pathInterval()(0),
                        m_path.pathInterval()(1)));
  Vector v = toppra::Vector::LinSpaced(0, 1, 10);
  path_parametrization = v;
  return 1;
};
} // namespace algorithm
} // namespace toppra
