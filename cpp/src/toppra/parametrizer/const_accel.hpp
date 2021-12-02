#ifndef TOPPRA_CONST_ACCEL_HPP
#define TOPPRA_CONST_ACCEL_HPP

#include <toppra/parametrizer.hpp>
#include <toppra/toppra.hpp>

namespace toppra {

namespace parametrizer {

/** \brief A path parametrizer with constant acceleration assumption.
 *
 * We assume that in each segment, the path acceleration is constant.
 */
class ConstAccel : public Parametrizer {
 public:
  ConstAccel(GeometricPathPtr path, const Vector &gridpoints, const Vector &vsquared);

  /** \brief Create a python file and execute it to plot the output parametrization and show it.
   *
   * \param[in] n_sample Number of samples.
   *
   */
  bool plot_parametrization(const int n_sample);

 private:
  /** Return joint derivatives at specified times. */
  Vectors eval_impl(const Vector &times, int order = 0) const override;
  bool validate_impl() const override;
  Bound pathInterval_impl() const override;
  void writeVectorToFile(std::ofstream& myfile, const Vector& vector, const std::string & name) const;
  void writeVectorsToFile(std::ofstream& myfile, const Vectors& vectors, const std::string & name) const;

  /** \brief Evaluate path variables ss, vs, and us at the input time instances.
   *
   * For each t, a starting gridpoint will be selected. The selected
   * path position, velocity and acceleration is then used to compute
   * the corresponding output quantities.
   *
   * \param[in] ts Time instances to be evaluated, starting from zero seconds.
   * \param[out] ss Path positions at the given time instances.
   * \param[out] vs Path velocities at the given time instances.
   * \param[out] us Path accelerations at the given time instances.
   *
   */
  bool evalParams(const Vector &ts, Vector &ss, Vector &vs, Vector &us) const;
  // Compute times and acclerations from given data (path, velocities)
  void process_internals();
  // Vector of time instances (corresponded to gridpoints)
  Vector m_ts;
  // Vector of accelerations (corresponded to gridpoints). Should have size
  // shorter than m_ts and m_vs by 1.
  Vector m_us;
};
}  // namespace parametrizer

}  // namespace toppra

#endif
