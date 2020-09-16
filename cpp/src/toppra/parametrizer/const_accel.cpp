#include <c++/7/bits/c++config.h>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <stdexcept>
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

bool ConstAccel::evalParams(const Vector& ts, Vector& ss, Vector& vs,
                            Vector& us) const {
  ss.resize(ts.size());
  vs.resize(ts.size());
  us.resize(ts.size());
  int k_grid = 0;
  for (std::size_t i = 0; i < ts.size(); i++) {
    // find k_grid s.t m_gridpoints[k_grid] <= ss[i] < m_gridpoints[k_grid + 1]

    // reset the search back to the first gridpoint, only do once
    if (m_ts[k_grid] > ts[i]) {
      k_grid = 0;
    }

    // increment k_grid until the condition is satisfied
    while (k_grid < (m_ts.size() - 1)) {
      if (m_ts[k_grid] <= ts[i] && ts[i] < m_ts[k_grid + 1]) {
        break;
      } else if (k_grid == (m_ts.size() - 2)) {
        break;
      } else {
        k_grid++;
      }
    }

    // compute ss[i], vs[i] and us[i] using the k_grid segment.
    // extrapolate if ts[i] < m_ts[0] or ts[i] >= m_ts[m_ts.size() - 1].
    value_type dt = ts[i] - m_ts[k_grid];
    us[i] = m_us[k_grid];
    vs[i] = m_vs[k_grid] + dt * us[i];
    ss[i] = m_gridpoints[k_grid] + dt * m_vs[k_grid] + 0.5 * dt * dt * m_us[k_grid];
  }
  return true;
}

Vectors ConstAccel::eval_impl(const Vector& times, int order) const {
  Vector ss, vs, us;
  TOPPRA_LOG_DEBUG("Enter eval_impl");
  bool ret = evalParams(times, ss, vs, us);
  assert(ret);
  if (order == 0) {
    return m_path->eval(ss, 0);
  } else if (order == 1) {
    auto ps = m_path->eval(ss, 1);
    Vectors qd;
    qd.resize(ps.size());
    for (std::size_t i = 0; i < times.size(); i++) {
      qd[i] = ps[i] * vs[i];
    }
    return qd;
  } else if (order == 2) {
    auto ps = m_path->eval(ss, 1);
    auto pss = m_path->eval(ss, 2);
    Vectors qdd;
    qdd.resize(ps.size());
    for (std::size_t i = 0; i < times.size(); i++) {
      qdd[i] = pss[i] * vs[i] * vs[i] + ps[i] * us[i];
    }
    return qdd;
  } else {
    throw std::runtime_error("Order >= 3 is not supported.");
  }
};

bool ConstAccel::validate_impl() const {
  // Normalize w.r.t max velocity value
  auto vs_norm = m_vs / m_vs.maxCoeff();
  // check that there is no "zero" velocity in the middle of the path
  for (std::size_t i = 0; i < vs_norm.size() - 1; i++) {
    // check that there is no huge fluctuation
    if (std::abs(vs_norm[i + 1] - vs_norm[i]) > 0.1) {
      TOPPRA_LOG_DEBUG("Large variation between path velocities detected. v[i+1] - v[i] = " << vs_norm[i + 1] - vs_norm[i]);
      return false;
    }
    // check if i != 0
    if (i == 0) continue;
    if (vs_norm[i] < 1e-5) {
      TOPPRA_LOG_DEBUG(
          "Very small path velocity detected. Probably a numerical error occured.");
      return false;
    }
  }
  return true;
}

Bound ConstAccel::pathInterval_impl() const {
  Bound b;
  b << m_gridpoints[0], m_gridpoints[m_gridpoints.size() - 1];
  return b;
}

}  // namespace parametrizer
}  // namespace toppra
