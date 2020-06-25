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
  ConstAccel(GeometricPathPtr path, const Vector &gridpoints, const Vector &vsquared);

 private:
  Vectors eval_impl(const Vector &, int order = 0) const override;
  bool validate_impl() const override;
  Bound pathInterval_impl() const override;

  // Compute times and acclerations from given data (path, velocities)
  void process_internals();
  // Vector of time instances (corresponded to gridpoints)
  Vector m_ts;
  // Vector of accelerations (corresponded to gridpoints). Should be
  // shorter than others by 1.
  Vector m_us;
};
}  // namespace parametrizer

}  // namespace toppra

#endif
