#ifndef TOPPRA_PARAMETRIZER_HPP
#define TOPPRA_PARAMETRIZER_HPP

#include <toppra/geometric_path.hpp>
#include <toppra/toppra.hpp>

namespace toppra {

/**
 * \brief Abstract output trajectory parametrizers.
 *
 * A parametrizer has the same interface as a geometric path. It
 * receives as input the original geometric path and two arrays of
 * gridpoints and parametrization (i.e. squared path velocities). A
 * parametrizer should validate the input data. If not validated,
 * evaluation results are not defined.
 *
 * Requirements: https://github.com/hungpham2511/toppra/issues/102
 *
 * Sub-classes should override the virtual private methods *_impl.
 */
class Parametrizer : public GeometricPath {
 public:
  /** Construct the parametrizer.
   *
   * \param path Input geometric path.
   * \param gridpoints Shape (N+1,). Gridpoints of the parametrization, should be
   * compatible with the path domain. \param vsquared Shape (N+1,). Path velocity
   * squared, should have same shape as gridpoints as vsquared[i] corresponds to the
   * velocity at gridpoints[i].
   */
  Parametrizer(GeometricPathPtr path, const Vector &gridpoints, const Vector &vsquared);

  /** \brief Evaluate the path at given position.
   */
  Vector eval_single(value_type, int order = 0) const override;

  /** \brief Evaluate the path at given positions (vector).
   */
  Vectors eval(const Vector &, int order = 0) const override;

  /** \brief Return the starting and ending path positions.
   */
  Bound pathInterval() const override;

  /** \brief Validate input data.
   *
   * Return false if something is wrong with the output
   * trajectory. Only use the trajectory if validation successes.
   */
  bool validate() const;

  virtual ~Parametrizer() {}

 protected:
  // Input geometric path
  GeometricPathPtr m_path;
  // Input gridpoints
  Vector m_gridpoints;
  // Input path velocities (not squared)
  Vector m_vs;

 private:
  /// To be overwriten by derived classes
  virtual Vectors eval_impl(const Vector &, int order = 0) const = 0;
  virtual bool validate_impl() const = 0;
  virtual Bound pathInterval_impl() const = 0;
};
};  // namespace toppra

#endif
