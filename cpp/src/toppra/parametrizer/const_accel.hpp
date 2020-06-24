#ifndef TOPPRA_CONST_ACCEL_HPP
#define TOPPRA_CONST_ACCEL_HPP

#include <toppra/parametrizer.hpp>
#include <toppra/toppra.hpp>

namespace toppra {

namespace parametrizer {

/**
 * \brief A path parametrizer with constant acceleration assumption.
 */
class ConstAccel : public Parametrizer {
 public:
  ConstAccel() = default;
  ConstAccel(GeometricPathPtr path, const Vector &gridpoints, const Vector &vsquared);

  /**
   * /brief Evaluate the path at given position.
   */
  Vector eval_single(value_type, int order = 0) const override;

  /**
   * /brief Evaluate the path at given positions (vector).
   */
  Vectors eval(const Vector &, int order = 0) const override;

  /**
   * Return the starting and ending path positions.
   */
  Bound pathInterval() const override;

 private:
  // Compute times and acclerations from given data (path, velocities)
  void process_parametrization();

  GeometricPathPtr m_path;
  // User given gridpoints
  Vector m_gridpoints;
  // Vector of path velocities (not squared)
  Vector m_vs;
  // Vector of time instances (corresponded to gridpoints)
  Vector m_ts;
  // Vector of accelerations (corresponded to gridpoints). Should be shorter than others
  // by 1.
  Vector m_us;
};
}  // namespace parametrizer

}  // namespace toppra

#endif
