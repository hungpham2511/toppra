#include <toppra/geometric_path/piecewise_poly_path.hpp>

#include <cstddef>
#include <stdexcept>
#include <ostream>
#include "toppra/toppra.hpp"

namespace toppra {

Matrix differentiateCoefficients(const Matrix &coefficients) {
  Matrix deriv(coefficients.rows(), coefficients.cols());
  deriv.setZero();
  for (size_t i = 1; i < coefficients.rows(); i++) {
    deriv.row(i) = coefficients.row(i - 1) * (coefficients.rows() - i);
  }
  return deriv;
}

PiecewisePolyPath::PiecewisePolyPath(const Matrices & coefficients,
                                     std::vector<value_type> breakpoints)
    : GeometricPath (coefficients[0].cols()),
      m_coefficients(coefficients), m_breakpoints(std::move(breakpoints)),
      m_degree(coefficients[0].rows() - 1) {

  checkInputArgs();
  computeDerivativesCoefficients();
}

Bound PiecewisePolyPath::pathInterval() const {
  Bound v;
  v << m_breakpoints.front(), m_breakpoints.back();
  return v;
};

Vector PiecewisePolyPath::eval_single(value_type pos, int order) const {
  assert(order < 3 && order >= 0);
  Vector v(m_dof);
  v.setZero();
  size_t seg_index = findSegmentIndex(pos);
  auto coeff = getCoefficient(seg_index, order);
  for (int power = 0; power < m_degree + 1; power++) {
    v += coeff.row(power) *
         pow(pos - m_breakpoints[seg_index], m_degree - power);
  }
  return v;
}

// Not the most efficient implementation. Coefficients are
// recompoted. Should be refactorred.
Vectors PiecewisePolyPath::eval(const Vector &positions, int order) const {
  assert(order < 3 && order >= 0);
  Vectors outputs;
  outputs.resize(positions.size());
  for (size_t i = 0; i < positions.size(); i++) {
    outputs[i] = eval_single(positions(i), order);
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
    std::ostringstream oss;
    oss << "Position " << pos << " is outside of range [ " << m_breakpoints[0]
      << ", " << m_breakpoints[m_breakpoints.size()-1] << ']';
    throw std::runtime_error(oss.str());
  }
  return seg_index;
}

void PiecewisePolyPath::checkInputArgs() {
  if ((1 + m_coefficients.size()) != m_breakpoints.size()) {
    throw std::runtime_error(
        "Number of breakpoints must equals number of segments plus 1.");
  }
  for (size_t seg_index = 0; seg_index < m_coefficients.size(); seg_index++) {
    if (m_breakpoints[seg_index] >= m_breakpoints[seg_index + 1]) {
      throw std::runtime_error("Require strictly increasing breakpoints");
    }
  }
}

void PiecewisePolyPath::computeDerivativesCoefficients() {
  m_coefficients_1.reserve(m_coefficients.size());
  m_coefficients_2.reserve(m_coefficients.size());
  for (size_t seg_index = 0; seg_index < m_coefficients.size(); seg_index++) {
    m_coefficients_1.push_back(
        differentiateCoefficients(m_coefficients[seg_index]));
    m_coefficients_2.push_back(
        differentiateCoefficients(m_coefficients_1[seg_index]));
  }
}

const Matrix& PiecewisePolyPath::getCoefficient(int seg_index, int order) const {
  if (order == 0) {
    return m_coefficients.at(seg_index);
  } else if (order == 1) {
    return m_coefficients_1.at(seg_index);
  } else if (order == 2) {
    return m_coefficients_2.at(seg_index);
  } else {
    return m_coefficients_2.at(seg_index);
  }
}

} // namespace toppra
