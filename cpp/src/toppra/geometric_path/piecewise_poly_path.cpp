#include <toppra/geometric_path/piecewise_poly_path.hpp>

#include <iostream>
#include <iomanip>
#include <ostream>
#include <stdexcept>
#include <string>
#include <array>

#include <Eigen/Dense>
#include <toppra/toppra.hpp>

#ifdef TOPPRA_OPT_MSGPACK
#include <msgpack.hpp>
#endif

namespace toppra {

BoundaryCond::BoundaryCond(int order, const std::vector<value_type> &values_in)
    : order(order), bc_type(Manual) {
  values.resize(values_in.size());
  for (std::size_t i = 0; i < values.size(); i++)
    values(i) = values_in[i];
}

BoundaryCond::BoundaryCond(int order, const Vector values): order(order), values(values), bc_type(Manual){};

BoundaryCond::BoundaryCond(std::string type){
  if (type == "not-a-knot" || type == "notaknot"){
    bc_type = Type::NotAKnot;
  } else if (type == "clamped") {
    bc_type = Type::Clamped;
  } else if (type == "natural") {
    bc_type = Type::Natural;
  } else
    throw std::runtime_error("Unknown bc type" + type);
}


namespace _cubic_spline{
/**
 * @brief Check, potentially modify input.
 * 
 * @param positions 
 * @param times 
 * @param bc_type 
 */
void checkInputArgs(const Vectors &positions, const Vector &times,
                    BoundaryCondFull &bc_type) {
  if (positions.size() != times.rows()) {
    throw std::runtime_error(
        "The length of 'positions' doesn't match the length of 'times'.");
  }
  if (times.rows() < 2) {
    throw std::runtime_error("'times' must contain at least 2 elements.");
  }
  for (size_t i = 1; i < positions.size(); i++) {
    if (positions[i].rows() != positions[i - 1].rows()) {
      throw std::runtime_error(
          "The number of elements in each position has to be equal.");
    }
  }
  Vector dtimes(times.rows() - 1);
  for (auto i = 1u; i < times.rows(); i++) {
    dtimes(i - 1) = times(i) - times(i - 1);
    if (dtimes(i - 1) <= 0) {
      throw std::runtime_error(
          "'times' must be a strictly increasing sequence.");
    }
  }

  // Validate boundary conditions
  size_t expected_deriv_size = positions[0].size();
  for (BoundaryCond &bc : bc_type) {
    if (bc.bc_type == BoundaryCond::Clamped){
      bc.order = 1;
      bc.values.resize(expected_deriv_size);
      bc.values.setZero();
    } else if (bc.bc_type == BoundaryCond::Natural){
      bc.order = 2;
      bc.values.resize(expected_deriv_size);
      bc.values.setZero();
    }  else if (bc.bc_type == BoundaryCond::NotAKnot) {
      throw std::runtime_error("Boundary condition NotAKnot is not implemented");
    } else if (bc.order != 1 && bc.order != 2) {
      throw std::runtime_error(
          "The specified derivative order must be 1 or 2.");
    } else if (bc.values.size() != expected_deriv_size) {
      throw std::runtime_error("`deriv_value` size " +
                               std::to_string(bc.values.size()) +
                               " is not the expected one " +
                               std::to_string(expected_deriv_size) + ".");
    }
  }
}

/**
 * @brief Compute the piecewise poly coefficients.
 * 
 * @param positions 
 * @param times 
 * @param bc_type 
 * @param coefficients 
 */
void computePPolyCoefficients(const Vectors &positions, const Vector &times,
                              const BoundaryCondFull &bc_type,
                              Matrices &coefficients) {
  // h(i) = t(i+1) - t(i)
  Vector h(times.rows() - 1);
  for (auto i = 0u; i < h.rows(); i++) {
    h(i) = times(i + 1) - times(i);
  }

  // Construct the tri-diagonal matrix A based on spline continuity criteria
  Matrix A = Matrix::Zero(times.rows(), times.rows());
  for (auto i = 1u; i < A.rows() - 1; i++) {
    A.row(i).segment(i - 1, 3) << h(i - 1), 2 * (h(i - 1) + h(i)), h(i);
  }

  // Construct B based on spline continuity criteria
  Vectors B(positions.at(0).rows());
  for (auto i = 0u; i < B.size(); i++) {
    B[i].resize(times.rows());
    for (auto j = 1u; j < A.rows() - 1; j++) {
      B[i](j) = 3 * (positions[j + 1](i) - positions[j](i)) / h(j) -
                3 * (positions[j](i) - positions[j - 1](i)) / h(j - 1);
    }
  }

  // Insert boundary conditions to A and B
  if (bc_type[0].order == 1) {
    A.row(0).segment(0, 2) << 2 * h(0), h(0);
    for (auto i = 0u; i < B.size(); i++) {
      B[i](0) = 3 * (positions[1](i) - positions[0](i)) / h(0) -
                3 * bc_type[0].values(i);
    }
  } else if (bc_type[0].order == 2) {
    A(0, 0) = 2;
    for (auto i = 0u; i < B.size(); i++) {
      B[i](0) = bc_type[0].values(i);
    }
  }

  if (bc_type[1].order == 1) {
    A.row(A.rows() - 1).segment(A.cols() - 2, 2) << h(h.rows() - 1),
        2 * h(h.rows() - 1);
    for (auto i = 0u; i < B.size(); i++) {
      B[i](B[i].rows() - 1) =
          3 * bc_type[1].values(i) - 3 *
                                         (positions[positions.size() - 1](i) -
                                          positions[positions.size() - 2](i)) /
                                         h(h.rows() - 1);
    }
  } else if (bc_type[1].order == 2) {
    A(A.rows() - 1, A.cols() - 1) = 2;
    for (auto i = 0u; i < B.size(); i++) {
      B[i](B[i].rows() - 1) = bc_type[1].values(i);
    }
  }

  // Solve AX = B
  Vectors X(positions[0].rows());
  for (auto i = 0u; i < X.size(); i++) {
    X[i].resize(times.rows());
    X[i] = A.colPivHouseholderQr().solve(B[i]);
  }

  // Insert spline coefficients
  for (auto i = 0u; i < coefficients.size(); i++) {
    coefficients[i].resize(4, positions[0].rows());
    for (auto j = 0u; j < coefficients[i].cols(); j++) {
      coefficients[i](0, j) = (X[j](i + 1) - X[j](i)) / (3 * h(i));
      coefficients[i](1, j) = X[j](i);
      coefficients[i](2, j) = (positions[i + 1](j) - positions[i](j)) / h(i) -
                              h(i) / 3 * (2 * X[j](i) + X[j](i + 1));
      coefficients[i](3, j) = positions[i](j);
    }
  }
}
}

PiecewisePolyPath
PiecewisePolyPath::CubicSpline(const Vectors &positions, const Vector &times,
                               BoundaryCondFull bc_type) {
  Matrices coefficients(times.size() - 1);
  _cubic_spline::checkInputArgs(positions, times, bc_type);
  _cubic_spline::computePPolyCoefficients(positions, times, bc_type, coefficients);
  std::vector<value_type> breakpoints(times.data(), times.data() + times.size());
  return PiecewisePolyPath(coefficients, breakpoints);
}


Matrix differentiateCoefficients(const Matrix &coefficients) {
  Matrix deriv(coefficients.rows(), coefficients.cols());
  deriv.setZero();
  for (Eigen::Index i = 1; i < coefficients.rows(); i++) {
    deriv.row(i) = coefficients.row(i - 1) * (coefficients.rows() - i);
  }
  return deriv;
}

PiecewisePolyPath::PiecewisePolyPath(const Matrices & coefficients,
                                     std::vector<value_type> breakpoints)
    : GeometricPath (int(coefficients[0].cols())),
      m_coefficients(coefficients), m_breakpoints(std::move(breakpoints)),
      m_degree(int(coefficients[0].rows()) - 1) {

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
// recomputed. Should be refactored.
Vectors PiecewisePolyPath::eval(const Vector &positions, int order) const {
  assert(order < 3 && order >= 0);
  Vectors outputs;
  outputs.resize(positions.size());
  for (auto i = 0u; i < positions.size(); i++) {
    outputs[i] = eval_single(positions(i), order);
  }
  return outputs;
}

size_t PiecewisePolyPath::findSegmentIndex(value_type pos) const {
  if (pos > m_breakpoints.back() || pos < m_breakpoints[0]) {
    constexpr auto max_precision {std::numeric_limits<value_type>::digits10 + 1};
    std::ostringstream oss;
    oss << std::setprecision(max_precision) << "Position " << pos <<
      " is outside of range [ " << m_breakpoints[0] << ", " << m_breakpoints.back() << ']';
    throw std::runtime_error(oss.str());
  }
  auto it = std::upper_bound(m_breakpoints.begin(), m_breakpoints.end(), pos);
  auto idx = std::distance(m_breakpoints.begin(), it) - 1;
  if(idx < 0)
    return 0;
  return std::min<size_t>(idx, m_coefficients.size() - 1);
}

void PiecewisePolyPath::checkInputArgs() {
  assert(m_coefficients[0].cols() == m_dof);
  assert(m_coefficients[0].rows() == (m_degree + 1));
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

const Matrix &PiecewisePolyPath::getCoefficient(size_t seg_index, int order) const {
  switch (order)
  {
    case 0: return m_coefficients.at(seg_index);
    case 1: return m_coefficients_1.at(seg_index);
    case 2: return m_coefficients_2.at(seg_index);
    default: break;
  }

  return m_coefficients_2.at(seg_index);
}

void PiecewisePolyPath::serialize(std::ostream &O) const {
#ifdef TOPPRA_OPT_MSGPACK
  MatricesData allraw;
  allraw.reserve(m_coefficients.size());
  for (const auto &c : m_coefficients) {
    MatrixData raw{c.rows(), c.cols(), {c.data(), c.data() + c.size()}};
    allraw.push_back(raw);
  }
  msgpack::pack(O, allraw);
  msgpack::pack(O, m_breakpoints);
#endif
}

void PiecewisePolyPath::deserialize(std::istream &I) {
#ifdef TOPPRA_OPT_MSGPACK
  std::stringstream buffer;
  buffer << I.rdbuf();
  std::size_t offset = 0;

  auto oh = msgpack::unpack(buffer.str().data(), buffer.str().size(), offset);
  auto obj = oh.get();
  TOPPRA_LOG_DEBUG(obj << "at offset:=" << offset << "/" << buffer.str().size());
  MatricesData x;
  toppra::Matrices new_coefficients;
  obj.convert(x);
  for (auto const &y : x) {
    int nrow, ncol;
    nrow = std::get<0>(y);
    ncol = std::get<1>(y);
    std::vector<value_type> mdata = std::get<2>(y);
    toppra::Matrix m(nrow, ncol);
    for (size_t i = 0; i < mdata.size(); i++) m(i) = mdata[i];
    TOPPRA_LOG_DEBUG(nrow << ncol << mdata.size() << m);
    new_coefficients.push_back(m);
  }

  reset();
  m_coefficients = new_coefficients;
  oh = msgpack::unpack(buffer.str().data(), buffer.str().size(), offset);
  obj = oh.get();
  TOPPRA_LOG_DEBUG(obj << "at offset:=" << offset << "/" << buffer.str().size());
  assert(offset == buffer.str().size());
  obj.convert(m_breakpoints);

  TOPPRA_LOG_DEBUG("degree: " << m_degree);
  m_dof = new_coefficients[0].cols();
  m_degree = new_coefficients[0].rows() - 1;
  checkInputArgs();
  computeDerivativesCoefficients();
#endif
}

void PiecewisePolyPath::reset() {
  m_breakpoints.clear();
  m_coefficients.clear();
  m_coefficients_1.clear();
  m_coefficients_2.clear();
}

void PiecewisePolyPath::initAsHermite(const Vectors &positions,
                                      const Vectors &velocities,
                                      const std::vector<value_type> times) {
  reset();
  assert(positions.size() == times.size());
  assert(velocities.size() == times.size());
  TOPPRA_LOG_DEBUG("Constructing new Hermite polynomial");
  m_configSize = m_dof = int(positions[0].size());
  m_degree = 3;  // cubic spline
  m_breakpoints = times;
  for (std::size_t i = 0; i < times.size() - 1; i++) {
    TOPPRA_LOG_DEBUG("Processing segment index: " << i << "/" << times.size() - 1);
    Matrix c(4, m_dof);
    auto dt = times[i + 1] - times[i];
    assert(dt > 0);
    // ... after some derivations
    c.row(3) = positions.at(i);
    c.row(2) = velocities.at(i);
    c.row(0) = (velocities.at(i + 1).transpose() * dt -
                2 * positions.at(i + 1).transpose() + c.row(2) * dt + 2 * c.row(3)) /
               pow(dt, 3);
    c.row(1) = (velocities.at(i + 1).transpose() - c.row(2) - 3 * c.row(0) * dt * dt) /
               (2 * dt);
    m_coefficients.push_back(c);
  }
  checkInputArgs();
  computeDerivativesCoefficients();
}

PiecewisePolyPath PiecewisePolyPath::constructHermite(
    const Vectors &positions, const Vectors &velocities,
    const std::vector<value_type> times) {
  PiecewisePolyPath path;
  path.initAsHermite(positions, velocities, times);
  return path;
}

PiecewisePolyPath PiecewisePolyPath::CubicHermiteSpline(
    const Vectors &positions, const Vectors &velocities,
    const std::vector<value_type> times) {
  PiecewisePolyPath path;
  path.initAsHermite(positions, velocities, times);
  return path;
}


} // namespace toppra
