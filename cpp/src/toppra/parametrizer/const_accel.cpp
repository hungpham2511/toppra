#include <algorithm>
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
  TOPPRA_LOG_DEBUG("process_internals: ts=" << m_ts.transpose());
  TOPPRA_LOG_DEBUG("process_internals: us=" << m_us.transpose());
  TOPPRA_LOG_DEBUG("process_internals: vs=" << m_vs.transpose());
}

bool ConstAccel::evalParams(const Vector& ts, Vector& ss, Vector& vs,
                            Vector& us) const {
  TOPPRA_LOG_DEBUG("evalParams: ts=" << ts.transpose());
  assert (m_ts.size() >= 2);
  ss.resize(ts.size());
  vs.resize(ts.size());
  us.resize(ts.size());
  int k_grid = 0;
  for (std::size_t i = 0; i < ts.size(); i++) {
    // find k_grid s.t                     m_ts[k_grid] <= ts[i] < m_ts[k_grid + 1], or
    //      k_grid = 0 if                  ts[i] < m_ts[0], or
    //      k_grid = m_ts.size() - 2 if    ts[i] > m_ts
    auto ptr = std::lower_bound(m_ts.data() + 1, m_ts.data() + m_ts.size() - 1, ts[i]);
    k_grid = ptr - m_ts.data() - 1;
    TOPPRA_LOG_DEBUG("ts[i]=" << ts[i] << ", k_grid=" << k_grid);

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
  TOPPRA_LOG_DEBUG("eval_impl. order=" << order);
  bool ret = evalParams(times, ss, vs, us);
  assert(ret);
  switch (order) {
    case 0:
      return m_path->eval(ss, 0);
      break;
    case 1: {
      auto ps = m_path->eval(ss, 1);
      Vectors qd;
      qd.resize(ps.size());
      for (std::size_t i = 0; i < times.size(); i++) {
        qd[i] = ps[i] * vs[i];
      }
      return qd;
    }
    case 2: {
      auto ps = m_path->eval(ss, 1);
      auto pss = m_path->eval(ss, 2);
      Vectors qdd;
      qdd.resize(ps.size());
      for (std::size_t i = 0; i < times.size(); i++) {
        qdd[i] = pss[i] * vs[i] * vs[i] + ps[i] * us[i];
      }
      return qdd;
    }
    default:
      throw std::runtime_error("Order >= 3 is not supported.");
  }
};

bool ConstAccel::validate_impl() const {
  // Normalize w.r.t max velocity value
  auto vs_norm = m_vs / m_vs.maxCoeff();
  // check that there is no "zero" velocity in the middle of the path
  for (std::size_t i = 0; i < vs_norm.size() - 1; i++) {
    // check that there is no huge fluctuation
    if (std::abs(vs_norm[i + 1] - vs_norm[i]) > 1) {
      TOPPRA_LOG_DEBUG(
          "Large variation between path velocities detected. v[i+1] - v[i] = "
          << vs_norm[i + 1] - vs_norm[i]);
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
  b << m_ts[0], m_ts[m_ts.size() - 1];
  return b;
}

}  // namespace parametrizer
}  // namespace toppra
