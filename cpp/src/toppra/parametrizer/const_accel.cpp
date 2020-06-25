#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <toppra/parametrizer.hpp>
#include <toppra/parametrizer/const_accel.hpp>
#include <toppra/toppra.hpp>

namespace toppra {
namespace parametrizer {

ConstAccel::ConstAccel(GeometricPathPtr path, const Vector& gridpoints,
                       const Vector& vsquared)
    : Parametrizer(path, gridpoints, vsquared) {
  process_internals();
}

void ConstAccel::process_internals() {
  m_ts.resize(m_gridpoints.size());
  m_us.resize(m_gridpoints.size() - 1);
  m_ts[0] = 0;
  for (std::size_t i = 0; i < m_gridpoints.size() - 1; i++) {
    m_us[i] = 0.5 * (m_vs[i + 1] * m_vs[i + 1] - m_vs[i] * m_vs[i]) /
              (m_gridpoints[i + 1] - m_gridpoints[i]);
    m_ts[i + 1] =
        m_ts[i] + 2 * (m_gridpoints[i + 1] - m_gridpoints[i]) / (m_vs[i + 1] + m_vs[i]);
  }
}

Vectors ConstAccel::eval_impl(const Vector& times, int order) const {
  return m_path->eval(times, order);
};

bool ConstAccel::validate_impl() const { return true; }

Bound ConstAccel::pathInterval_impl() const {
  Bound b;
  b << m_gridpoints[0], m_gridpoints[m_gridpoints.size() - 1];
  return b;
}

}  // namespace parametrizer
}  // namespace toppra
