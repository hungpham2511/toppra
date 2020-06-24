#include <cmath>
#include <cstdlib>
#include <cstddef>
#include <toppra/parametrizer/const_accel.hpp>
#include <toppra/toppra.hpp>


namespace toppra {
namespace parametrizer {

ConstAccel::ConstAccel(GeometricPathPtr path, const Vector & gridpoints, const Vector & vsquared):
    m_path(path), m_gridpoints(gridpoints) {
  assert(gridpoints.size() == vsquared.size());
  m_vs.resize(vsquared.size());
  for (std::size_t i=0; i < gridpoints.size(); i++){
    assert(vsquared[i] >= 0);
    m_vs[i] = std::sqrt(vsquared[i]);
    if (i == 0) continue;
    assert(gridpoints[i + 1] > gridpoints[i]);
  }

  assert(std::abs(path->pathInterval()[0] - gridpoints[0]) < TOPPRA_NEARLY_ZERO);
  assert(std::abs(path->pathInterval()[1] - gridpoints[gridpoints.size() - 1]) < TOPPRA_NEARLY_ZERO);
  
  process_parametrization();
}

void ConstAccel::process_parametrization(){
  m_ts.resize(m_gridpoints.size());
  m_us.resize(m_gridpoints.size() - 1);
  m_ts[0] = 0;
  for (std::size_t i=0; i < m_gridpoints.size() - 1; i++){
    m_us[i] = 0.5 * (m_vs[i + 1] * m_vs[i + 1] - m_vs[i] * m_vs[i]) / (m_gridpoints[i + 1] - m_gridpoints[i]);
    m_ts[i+1] = m_ts[i] + 2 * (m_gridpoints[i + 1] - m_gridpoints[i]) / (m_vs[i + 1] + m_vs[i]);
  }
}

// wrong impl
Vector ConstAccel::eval_single(value_type val, int order) const {
  return m_path->eval_single(val, order);
}

// wrong impl
Vectors ConstAccel::eval(const Vector & positions, int order) const {
  return m_path->eval(positions, order);
};


Bound ConstAccel::pathInterval() const {
  Bound b;
  b << m_gridpoints[0], m_gridpoints[m_gridpoints.size() - 1];
  return b;
}


}
}
