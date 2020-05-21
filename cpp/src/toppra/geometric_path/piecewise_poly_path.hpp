#ifndef TOPPRA_PIECEWISE_POLY_PATH_HPP
#define TOPPRA_PIECEWISE_POLY_PATH_HPP

#include <toppra/toppra.hpp>
#include <toppra/geometric_path.hpp>


namespace toppra {

/**
 * \brief Piecewise polynomial geometric path.
 *
 * A simple implemetation of a piecewise polynoamial geometric path.
 *
 */
class PiecewisePolyPath : public GeometricPath {
public:

  PiecewisePolyPath() = default;

  /**
   * Consructor.
   *
   * @param coefficients Polynoamial coefficients.
   * @param breakpoints Vector of breakpoints.
   */
  PiecewisePolyPath(const Matrices& coefficients, std::vector<value_type> breakpoints);

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

 private:
  size_t findSegmentIndex(value_type pos) const;
  void checkInputArgs();
  void computeDerivativesCoefficients();
  const Matrix& getCoefficient(int seg_index, int order) const;
  Matrices m_coefficients, m_coefficients_1, m_coefficients_2;
  std::vector<value_type> m_breakpoints;
  int m_degree;
};

}


#endif
