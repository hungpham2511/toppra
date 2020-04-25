#ifndef TOPPRA_GEOMETRIC_PATH_HPP
#define TOPPRA_GEOMETRIC_PATH_HPP

#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <toppra/toppra.hpp>
#include <vector>

namespace toppra {


/**
 * \brief Abstract interface for geometric paths.
 */
class GeometricPath {
public:

  GeometricPath()  = default;

  /**
   * Constructor of GeometricPath on vector spaces.
   */
  GeometricPath(int nDof) : m_configSize(nDof), m_dof (nDof) {}

  /**
   * Constructor of GeometricPath on non-vector spaces.
   */
  GeometricPath(int configSize, int nDof) : m_configSize(configSize), m_dof (nDof) {}

  /**
   * /brief Evaluate the path at given position.
   */
  virtual Vector eval_single(value_type, int order = 0) const = 0;

  /**
   * /brief Evaluate the path at given positions (vector).
   *
   * Default implementation: Evaluation each point one-by-one.
   */
  virtual Vectors eval(const Vector &positions, int order = 0) const;

  /**
   * \return the dimension of the configuration space
   */
  int configSize() const
  {
    return m_configSize;
  }

  /**
   * \return the number of degrees-of-freedom of the path.
   */
  int dof() const
  {
    return m_dof;
  }

  /**
   * \return the starting and ending path positions.
   */
  virtual Bound pathInterval() const = 0;

  virtual ~GeometricPath () {}

protected:
  int m_configSize, m_dof;
};

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
  PiecewisePolyPath(const Matrices &, const std::vector<value_type> &);

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
  Matrix getCoefficient(int seg_index, int order) const;
  Matrices m_coefficients, m_coefficients_1, m_coefficients_2;
  std::vector<value_type> m_breakpoints;
  int m_degree;
};

} // namespace toppra

#endif
