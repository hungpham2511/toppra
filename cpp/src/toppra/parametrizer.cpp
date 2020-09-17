#include <toppra/parametrizer.hpp>
#include <toppra/toppra.hpp>

namespace toppra {

Parametrizer::Parametrizer(GeometricPathPtr path, const Vector& gridpoints,
                           const Vector& vsquared)
    : m_path(path), m_gridpoints(gridpoints) {
  assert(gridpoints.size() == vsquared.size());
  m_vs.resize(vsquared.size());
  for (std::size_t i = 0; i < gridpoints.size(); i++) {
    assert(vsquared[i] >= 0);
    m_vs[i] = std::sqrt(vsquared[i]);
    if (i == gridpoints.size() - 1) continue;
    assert(gridpoints[i + 1] > gridpoints[i]);
  }

  assert(std::abs(path->pathInterval()[0] - gridpoints[0]) < TOPPRA_NEARLY_ZERO);
  assert(std::abs(path->pathInterval()[1] - gridpoints[gridpoints.size() - 1]) <
         TOPPRA_NEARLY_ZERO);
}

Vector Parametrizer::eval_single(value_type val, int order) const {
  Vector v{1};
  v << val;
  auto results = eval_impl(v, order);
  return results[0];
}

Vectors Parametrizer::eval(const Vector& positions, int order) const {
  return eval_impl(positions, order);
}

Bound Parametrizer::pathInterval() const { return pathInterval_impl(); }

bool Parametrizer::validate() const { return validate_impl(); }
}  // namespace toppra
