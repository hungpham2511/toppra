#ifndef TOPPRA_GEOMETRIC_PATH_HPP
#define TOPPRA_GEOMETRIC_PATH_HPP

#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <toppra/toppra.hpp>
#include <vector>

namespace toppra {
class GeometricPath {};

Matrix differentiateCoefficients(const Matrix &coefficients) {
  Matrix deriv(coefficients.rows(), coefficients.cols());
  deriv.setZero();
  for (size_t i=1; i < coefficients.rows(); i++){
    deriv.row(i) = coefficients.row(i - 1) * (coefficients.rows() - i);
  }
  return deriv;
}

/**
 * Piecewise polynomial geometric path.
 */
class PiecewisePolyPath : public GeometricPath {
public:
  PiecewisePolyPath(const Matrices &, const std::vector<value_type> &);
  Vector eval(value_type, int degree = 0);

private:
  Matrices m_coefficients;
  std::vector<value_type> m_breakpoints;
  int m_dof, m_degree;
};

PiecewisePolyPath::PiecewisePolyPath(const Matrices &coefficients,
                                     const std::vector<value_type> &breakpoints)
    : m_coefficients(coefficients), m_breakpoints(breakpoints),
      m_dof(coefficients[0].cols()), m_degree(coefficients[0].rows() - 1) {}

Vector PiecewisePolyPath::eval(value_type pos, int order) {
  Vector v(m_dof);
  v.setZero();
  std::cout << m_coefficients[0].row(0);
  size_t seg_index = -1;
  for (size_t i = 0; i < m_coefficients.size(); i++) {
    if (m_breakpoints[i] <= pos && pos <= m_breakpoints[i + 1]) {
      seg_index = i;
      break;
    }
  }
  if (seg_index == -1) {
    throw std::runtime_error("Given position is outside of breakpoints' range");
  }

  Matrix coeff = m_coefficients[seg_index];
  for (size_t i=0; i < order; i++){
    coeff = differentiateCoefficients(coeff);
  }
  for (int power = 0; power < m_degree + 1; power++) {
    v += coeff.row(power) * pow(pos - m_breakpoints[seg_index], m_degree - power);
  }
  return v;
} // namespace toppra



} // namespace toppra

#endif
