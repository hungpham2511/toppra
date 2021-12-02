#include <algorithm>
#include <fstream>
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

  // Check that the requested evaluation times are within
  // bounds. Small tolerances (based on TOPPRA_ABS_TOL and
  // TOPPRA_REL_TOL) are allowed. This quantity is small (~1e-8 sec)
  // so it has almost nil effect on the physical trajectory quality.
  auto interval = pathInterval_impl();
  if ((times.array() < (interval[0] * (1 - TOPPRA_REL_TOL) - TOPPRA_ABS_TOL)).any()) {
    throw std::runtime_error("Request time instances out of numerical lower bounds.");
  }
  else if ((times.array() > (interval[1] * (1 + TOPPRA_REL_TOL) + TOPPRA_ABS_TOL)).any())
  {
    throw std::runtime_error("Request time instances out of numerical upper bounds.");
  }

  // Even if the times are within the bounds of the parametrizer, numerical
  // errors can still push our values outside the bounds of the path, so
  // we clamp values of ss to the path interval
  auto path_interval = m_path->pathInterval();
  ss = ss.cwiseMax(path_interval[0]).cwiseMin(path_interval[1]);

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

bool ConstAccel::plot_parametrization(const int n_sample) {
  // reimplements the function plot_parametrization() from the file toppra/parametrizer.py 
  Vector _ss = this->m_gridpoints;
  Vector _velocities = this->m_vsquared;
  Bound pi = this->pathInterval();
  Vector ts = Vector::LinSpaced(n_sample, pi(0), pi(1));
  Vector ss, vs, us;
  bool ok = this->evalParams(ts, ss, vs, us);
  if(!ok){
    return false;
  }
  Vectors qs = m_path->eval(ss);
  Vector ss_dense = Vector::LinSpaced(n_sample, _ss(0), _ss(_ss.size()-1));
  Vectors _ss_dense = m_path->eval(ss_dense);

  //write python code to the file
  std::ofstream myfile;
  myfile.open("plot_parametrization.py");
  myfile << "import numpy as np\n";
  myfile << "import matplotlib.pyplot as plt\n";
  writeVectorToFile(myfile, ts, "ts");
  writeVectorToFile(myfile, ss, "ss");
  writeVectorToFile(myfile, vs, "vs");
  writeVectorsToFile(myfile, qs, "qs");
  writeVectorToFile(myfile, this->m_ts, "_ts");
  writeVectorToFile(myfile, _ss, "_ss");
  writeVectorToFile(myfile, _velocities, "_velocities");
  writeVectorToFile(myfile, ss_dense, "ss_dense");
  writeVectorsToFile(myfile, _ss_dense, "_ss_dense");
  myfile << "plt.subplot(2, 2, 1)\n";
  myfile << "plt.plot(ts, ss, label=\"s(t)\")\n";
  myfile << "plt.plot(_ts, _ss, \"o\", label=\"input\")\n";
  myfile << "plt.title(\"path(time)\")\n";
  myfile << "plt.legend()\n";
  myfile << "plt.subplot(2, 2, 2)\n";
  myfile << "plt.plot(ss, vs, label=\"v(s)\")\n";
  myfile << "plt.plot(_ss, _velocities, \"o\", label=\"input\")\n";
  myfile << "plt.title(\"velocity(path)\")\n";
  myfile << "plt.legend()\n";
  myfile << "plt.subplot(2, 2, 3)\n";
  myfile << "plt.plot(ts, qs)\n";
  myfile << "plt.title(\"retimed path\")\n";
  myfile << "plt.subplot(2, 2, 4)\n";
  myfile << "plt.plot(ss_dense, _ss_dense)\n";
  myfile << "plt.title(\"original path\")\n";
  myfile << "plt.tight_layout()\n";
  myfile << "plt.show()\n";
  myfile.close();

  //execute python file
  system("python plot_parametrization.py");

  return true;
}

void ConstAccel::writeVectorToFile(std::ofstream& myfile, const Vector& vector, const std::string & name) const {
  const double factor = 10000;
  myfile << name << " = np.array([";
  for(size_t i=0; i<vector.size(); i++){
    myfile << factor*vector(i);
    if(i<vector.size()-1){
      myfile << ", ";
    }
  }
  myfile << "])\n";
  myfile << name << " = " << name << " / " << std::to_string(factor) << "\n";
}

void ConstAccel::writeVectorsToFile(std::ofstream& myfile, const Vectors& vectors, const std::string & name) const {
  const double factor = 10000;
  myfile << name << " = np.array([";
  for (size_t i = 0; i < vectors.size(); i++) {
    Vector vector = vectors.at(i);
    myfile << "[";
    for (size_t j = 0; j < vector.size(); j++) {
      myfile << factor*vector(j);
      if(j<vector.size()-1){
        myfile << ", ";
      }
    }
    myfile << "]";
    if(i<vectors.size()-1){
      myfile << ", ";
    }
  }
  myfile << "])\n";
  myfile << name << " = " << name << " / " << std::to_string(factor) << "\n";
}

}  // namespace parametrizer
}  // namespace toppra
