#ifndef TOPPRA_PIECEWISE_POLY_PATH_HPP
#define TOPPRA_PIECEWISE_POLY_PATH_HPP

#include <toppra/geometric_path.hpp>
#include <toppra/toppra.hpp>

namespace toppra {

/// Boundary condition
typedef struct {
    int order;
    Vector values;
} BoundaryCond;

/**
 * \brief Piecewise polynomial geometric path.
 *
 * An implementation of a piecewise polynomial geometric path.
 *
 * The coefficient vector has shape (N, P, D), where N is the number
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
   * @param coefficients Polynomial coefficients.
   * @param breakpoints Vector of breakpoints.
   */
  PiecewisePolyPath(const Matrices &coefficients, std::vector<value_type> breakpoints);

  /**
   * \brief Construct a new piecewise 3rd degree polynomial.
   * @param positions Vectors of path positions at given times.
   * @param times Vector of times in a strictly increasing order.
   * @param bc_type Boundary conditions at the curve start and end.
   */
  PiecewisePolyPath(const Vectors &positions, const Vector &times, const std::array<BoundaryCond, 2> &bc_type);

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

  /**
   * @brief Construct a piecewise Cubic Hermite polynomial.
   *
   * See https://en.wikipedia.org/wiki/Cubic_Hermite_spline for a good
   * description of this interplation scheme.
   *
   * This function is implemented based on scipy.interpolate.CubicHermiteSpline.
   *
   * Note that path generates by this function is not guaranteed to have
   * continuous acceleration, or the path second-order derivative.
   *
   * @param positions Robot joints corresponding to the given times. Must have
   * the same size as times.
   * @param velocities Robot joint velocities.
   * @param times Path positions or times. This is the independent variable.
   * @return PiecewisePolyPath
   */
  static PiecewisePolyPath constructHermite(const Vectors &positions,
                                            const Vectors &velocities,
                                            const std::vector<value_type> times);

  /**
   * @brief Construct a cubic spline.
   *
   * Interpolate the given joint positions with a spline that is twice
   * continuously differentiable. This means the position, velocity and
   * acceleration are guaranteed to be continous but not jerk.
   *
   * This method is modelled after scipy.interpolate.CubicSpline.
   *
   * @param positions Robot joints corresponding to the given times. 
   * @param times Path positions or times. This is the independent variable.
   * @param bc_type Boundary condition.
   * @return PiecewisePolyPath
   */
  static PiecewisePolyPath CubicSpline(const Vectors &positions, const Vector &times, const std::array<BoundaryCond, 2> &bc_type){
    return PiecewisePolyPath(positions, times, bc_type);
  }

 protected:
  void initAsHermite(const Vectors &positions, const Vectors &velocities,
                     const std::vector<value_type> times);
  void computeCubicSplineCoefficients(const Vectors &positions, const Vector &times,
          const std::array<BoundaryCond, 2> &bc_type, Matrices &coefficients);
  void reset();
  size_t findSegmentIndex(value_type pos) const;
  void checkInputArgs();
  void checkInputArgs(const Vectors &positions, const Vector &times,
                      const std::array<BoundaryCond, 2> &bc_type);
  void computeDerivativesCoefficients();
  const Matrix &getCoefficient(size_t seg_index, int order) const;
  Matrices m_coefficients, m_coefficients_1, m_coefficients_2;
  std::vector<value_type> m_breakpoints;
  int m_degree;
};

}  // namespace toppra

#endif
