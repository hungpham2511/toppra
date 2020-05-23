#ifndef TOPPRA_PIECEWISE_POLY_PATH_HPP
#define TOPPRA_PIECEWISE_POLY_PATH_HPP

#include <toppra/geometric_path.hpp>
#include <toppra/toppra.hpp>

namespace toppra {

/**
 * \brief Piecewise polynomial geometric path.
 *
 * An implemetation of a piecewise polynoamial geometric path.
 *
 * The coefficent vector has shape (N, P, D), where N is the number
 * segments. For each segment, the i-th row (P index) denotes the
 * power, while the j-th column is the degree of freedom. In
 * particular,
 *
 *  coeff(0) * dt ^ 3 + coeff(1) * dt ^ 2 + coeff(2) * dt + coeff(3)
 *
 *
 */
class PiecewisePolyPath : public GeometricPath {
 public:
  PiecewisePolyPath() = default;

  /**
   * \brief Construct new piecewise polynomial.
   *
   * See class docstring for details.
   *
   * @param coefficients Polynoamial coefficients.
   * @param breakpoints Vector of breakpoints.
   */
  PiecewisePolyPath(const Matrices &coefficients, std::vector<value_type> breakpoints);

  /**
   * /brief Evaluate the path at given position.
   */
  Vector eval_single(value_type, int order = 0) const;

  /**
   * /brief Evaluate the path at given positions (vector).
   */
  Vectors eval(const Vector &, int order = 0) const;

  /**
   * Return the starting and ending path positions.
   */
  Bound pathInterval() const;
  void serialize(std::ostream &O) const override;
  void deserialize(std::istream &I) override;

  void constructHermite(const Vectors& positions, const Vectors& velocities,
                        const std::vector<value_type> times) ;

 protected:
  void reset();
  size_t findSegmentIndex(value_type pos) const;
  void checkInputArgs();
  void computeDerivativesCoefficients();
  const Matrix &getCoefficient(int seg_index, int order) const;
  Matrices m_coefficients, m_coefficients_1, m_coefficients_2;
  std::vector<value_type> m_breakpoints;
  int m_degree;
};

}  // namespace toppra

#endif
