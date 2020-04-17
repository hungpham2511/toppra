#include "toppra/toppra.hpp"
#include <cstddef>
#include <toppra/geometric_path.hpp>

namespace toppra {

Matrix differentiateCoefficients(const Matrix &coefficients) {
  Matrix deriv(coefficients.rows(), coefficients.cols());
  deriv.setZero();
  for (size_t i = 1; i < coefficients.rows(); i++) {
    deriv.row(i) = coefficients.row(i - 1) * (coefficients.rows() - i);
  }
  return deriv;
}

PiecewisePolyPath::PiecewisePolyPath(const Matrices &coefficients,
                                     const std::vector<value_type> &breakpoints)
    : m_coefficients(coefficients), m_breakpoints(breakpoints),
      m_dof(coefficients[0].cols()), m_degree(coefficients[0].rows() - 1) {

  for(size_t seg_index =0; seg_index < m_coefficients.size(); seg_index++){
    m_coefficients_1.push_back(differentiateCoefficients(m_coefficients[seg_index]));
    m_coefficients_2.push_back(differentiateCoefficients(m_coefficients_1[seg_index]));
  }
  
}

Vector PiecewisePolyPath::eval(value_type pos, int order) {
  Vector v(m_dof);
  v.setZero();
  size_t seg_index = findSegmentIndex(pos);

  Matrix coeff;
  if (order == 0){
    coeff = m_coefficients[seg_index];
  } else if (order == 1) {
    coeff = m_coefficients_1[seg_index];
  } else if (order == 2) {
    coeff = m_coefficients_2[seg_index];
  }
  for (int power = 0; power < m_degree + 1; power++) {
    v += coeff.row(power) * pow(pos - m_breakpoints[seg_index], m_degree - power);
  }
  return v;
}


// Not the most efficient implementation. Coefficients are
// recompoted. Should be refactorred.
Vectors PiecewisePolyPath::eval(std::vector<value_type> positions, int order) {
  Vectors outputs;
  outputs.resize(positions.size());
  for (size_t i = 0; i < positions.size(); i++) {
    outputs[i] = eval(positions[i], order);
  }
  return outputs;
}

size_t PiecewisePolyPath::findSegmentIndex(value_type pos) const {
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
  return seg_index;
}

} // namespace toppra
