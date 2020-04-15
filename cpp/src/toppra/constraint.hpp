#ifndef TOPPRA_CONSTRAINT_HPP
#define TOPPRA_CONSTRAINT_HPP

#include <ostream>
#include <toppra/toppra.hpp>
#include <toppra/geometric_path.hpp>

namespace toppra {
/** Type of path parametrization constraint.
 * */
enum ConstraintType {
    Unknown         = 0,
    CanonicalLinear = 1,
    CanonicalConic  = 2,
};

/** Enum to mark different Discretization Scheme for LinearConstraint.
 *  In general, the difference in speed is not too large. Should use
 *  \ref Interpolation if possible.
 * */
enum DiscretizationType {
    Collocation,   /// smaller problem size, but lower accuracy.
    Interpolation, /// larger problem size, but higher accuracy.
};


/** The base constraint class.
 *
 *  Also known as Second-order Constraint.
 *
 *  A Canonical Linear Constraint has the following form:
 *  \f{eqnarray}
 *      \mathbf a_i u + \mathbf b_i x + \mathbf c_i &= v \\
 *      \mathbf F_i v & \leq \mathbf g_i \\
 *      x^{bound}_{i, 0} \leq x & \leq x^{bound}_{i, 1} \\
 *      u^{bound}_{i, 0} \leq u & \leq u^{bound}_{i, 1}
 *  \f}
 *
 *  Alternatively, if \f$ \mathbf F_i \f$ is constant for all values
 *  of \f$i\f$, then we can consider the simpler constraint:
 *  \f[
 *      \mathbf{F} v \leq \mathbf w
 *  \f]
 *
 *  In this case, the returned value of \f$F\f$ by
 *  LinearConstraint::computeParams has shape (k, m) instead of (N, k, m),
 *  \f$ w \f$ (k) instead of (N, k) and the class attribute
 *  LinearConstraint::identical will be \c true.
 *
 *  \note Derived classes should at least implement the method
 *  LinearConstraint::computeParams.
 *
 *  \sa JointAccelerationConstraint, JointVelocityConstraint,
 *  CanonicalLinearSecondOrderConstraint
 *
 * */
class LinearConstraint {
  public:
    ConstraintType constraintType () const
    {
      return constraintType_;
    }

    DiscretizationType discretizationType () const
    {
      return discretizationType_;
    }

    /** Tells whether the \f$ F \f$ matrix is the same over all the grid points.
     * In this case, LinearConstraint::computeParams F parameters should only
     * be of size 1.
     * */
    bool constantF () const
    {
      return constantF_;
    }

    void discretizationType (DiscretizationType type);

    /** Compute numerical coefficients of the given constraint.
     *
     *  \param[in] path The geometric path.
     *  \param[in] gridpoints Vector of size N+1. Gridpoint use for discretizing path.
     *
     *  \param[out] a N+1 Vector of size m.
     *  \param[out] b N+1 Vector of size m.
     *  \param[out] c N+1 Vector of size m.
     *  \param[out] F N+1 Matrix of shape (k, m). If LinearConstraint::constantF
     *              is \c true, there is only one such Matrix.
     *  \param[out] g N+1 Vector of size m.
     *  \param[out] ubound Shape (N + 1, 2). See notes.
     *  \param[out] xbound Shape (N + 1, 2). See notes.
     *
     * \note the output must be allocated to correct sizes prior to calling this
     * function.
     *
     * */
    void computeParams(const GeometricPath& path, const Vector& gridpoints,
        Vectors& a, Vectors& b, Vectors& c,
        Matrices& F, Vectors& g,
        Bounds& ubound, Bounds& xbound);

  protected:
    LinearConstraint()
      : constraintType_ (CanonicalLinear)
      , discretizationType_ (Collocation)
      , constantF_ (false)
    {}

    virtual std::ostream& print(std::ostream& os) const;
    virtual void computeParams_impl(const GeometricPath& path,
        const Vector& gridpoints,
        Vectors& a, Vectors& b, Vectors& c,
        Matrices& F, Vectors& g,
        Bounds& ubound, Bounds& xbound) = 0;

    ConstraintType constraintType_;
    DiscretizationType discretizationType_;
    bool constantF_;
}; // class LinearConstraint
} // namespace toppra

#endif
