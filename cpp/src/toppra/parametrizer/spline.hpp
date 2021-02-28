#ifndef TOPPRA_SPLINE_HPP
#define TOPPRA_SPLINE_HPP

#include <toppra/parametrizer.hpp>
#include <toppra/toppra.hpp>

namespace toppra {

namespace parametrizer {

/** \brief A path parametrizer via a spline interpolation.
 *
 * This class computes the time and position at each gridpoint, then
    fit and return a CubicSpline (continuous first and second
    derivatives). Note that the boundary conditions are: first
    derivatives at the start and end of the path equal q(s)' * s'.
 */
class Spline: public Parametrizer {
public:
    /**
     * \brief Construct the spline parametrizer.
     *
     * See class docstring for details.
     *
     * @param path the input geometric path.
     * @param gridpoints of the parametrization with shape (N+1,).
     * @param vsquared the path velocity squared with shape (N+1,).
    */
    Spline(GeometricPathPtr path, const Vector &gridpoints, const Vector &vsquared);
private:
    /** Return joint derivatives at specified times. */
    Vectors eval_impl(const Vector &times, int order = 0) const override;
    bool validate_impl() const override;
    Bound pathInterval_impl() const override;

    // Vector of time instances (corresponded to gridpoints)
    Vector m_ts;
};

}  // namespace parametrizer

}  // namespace toppra

#endif
