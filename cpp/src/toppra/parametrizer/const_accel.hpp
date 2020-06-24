#ifndef TOPPRA_CONST_ACCEL_HPP
#define TOPPRA_CONST_ACCEL_HPP

#include <toppra/toppra.hpp>
#include <toppra/parametrizer.hpp>

namespace toppra {

namespace parametrizer {

/**
 * \brief A path parametrizer with constant acceleration assumption.
 */
class ConstAccel: public Parametrizer {
 public:
  ConstAccel (GeometricPathPtr path, const Vector & gridpoints, const Vector & vsquared);

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
  void serialize(std::ostream &O) const override;
  void deserialize(std::istream &I) override;


};
}

}

#endif
