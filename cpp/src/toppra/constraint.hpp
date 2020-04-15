#ifndef TOPPRA_CONSTRAINT_HPP
#define TOPPRA_CONSTRAINT_HPP

#include <ostream>
#include <toppra/toppra.hpp>

namespace toppra {
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
 *      \mathbf F_i v & \leq \mathbf g_i
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
    DiscretizationType discretizationType () const
    {
      return discretizationType_;
    }

    void discretizationType (DiscretizationType type);

    /** Tells whether the \f$ F \f$ matrix is the same over all the grid points.
     * In this case, LinearConstraint::computeParams F parameters should only
     * be of size 1.
     * */
    bool constantF () const
    {
      return constantF_;
    }

    /// Dimension of \f$g\f$.
    Eigen::Index nbConstraints () const
    {
      return k_;
    }

    /// Dimension of \f$a, b, c, v\f$.
    Eigen::Index nbVariables () const
    {
      return m_;
    }

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
     * \todo check constness
     *
     * */
    void computeParams(const GeometricPath& path, const Vector& gridpoints,
        Vectors& a, Vectors& b, Vectors& c,
        Matrices& F, Vectors& g);

    virtual std::ostream& print(std::ostream& os) const;

    virtual ~LinearConstraint () {}

  protected:
    /**
     * \param k number of inequality constraints.
     * \param m number of internal variable (i.e. dimention of \f$v\f$).
     * */
    LinearConstraint(Eigen::Index k, Eigen::Index m)
      : discretizationType_ (Collocation)
      , k_ (k), m_ (m)
      , constantF_ (false)
    {}

    virtual void computeParams_impl(const GeometricPath& path,
        const Vector& gridpoints,
        Vectors& a, Vectors& b, Vectors& c,
        Matrices& F, Vectors& g) = 0;

    /// Convert from Collocation to Interpolation
    /// \todo in this case, the dimensions of the outputs is changed.
    void collocationToInterpolate (const Vector& gridpoints,
        Vectors& a, Vectors& b, Vectors& c,
        Matrices& F, Vectors& g);

    Eigen::Index k_, m_;
    DiscretizationType discretizationType_;
    bool constantF_;
}; // class LinearConstraint

/** BoxConstraint are LinearConstraint of the following form:
 *  \f{eqnarray}
 *      x^b_{i, 0} \leq x & \leq x^b_{i, 1} \\
 *      u^b_{i, 0} \leq u & \leq u^b_{i, 1}
 *  \f}
 *  where \f$x^b, u^b\f$ are the Bound vectors.
 *
 *  This class handles bounds on \f$x\f$, \f$u\f$ or both.
 *
 *  The LinearConstraint representation is computed as follows:
 *  \f{eqnarray}
 *      \mathbf a_i &= (1, -1, 0, 0).T \\
 *      \mathbf b_i &= (0, 0, 1, -1).T \\
 *      \mathbf c_i &= (-u^b_{i,1}, u^b_{i,0}, -x^b_{i,1}, x^b_{i,0}).T \\
 *      \mathbf F   &= I_4 \\
 *      \mathbf g_i &= 0_4 \\
 *  \f}
 *
 * */
class BoxConstraint : public LinearConstraint {
  public:
    /** Whether this constraint has bounds on \f$u\f$.
     * */
    bool hasUbounds () const
    {
      return hasUbounds_;
    }

    /** Whether this constraint has bounds on \f$x\f$.
     * */
    bool hasXbounds () const
    {
      return hasXbounds_;
    }

    /**
     * \todo check constness.
     * */
    void computeBounds(const GeometricPath& path, const Vector& gridpoint,
        Bounds& ubound, Bounds& xbound);

    virtual std::ostream& print(std::ostream& os) const;

    virtual ~BoxConstraint () {}

  protected:
    /**
     * \param uBound whether \f$u\f$ is bounded.
     * \param xBound whether \f$x\f$ is bounded.
     * */
    BoxConstraint(bool uBound, bool xBound)
      : LinearConstraint ((uBound && xBound) ? 4 : 2, (uBound && xBound) ? 4 : 2)
      , hasUbounds_ (uBound)
      , hasXbounds_ (xBound)
    {
      constantF_ = true;
      if (!hasXbounds_ && !hasUbounds_)
        std::invalid_argument("BoxConstraint must have bounds on X or U or both.");
    }

    /**
     * \todo at the moment, this methods throws but the matrices a, b, c, F and g
     * could be easily computed from the bounds.
     * */
    void computeParams_impl(const GeometricPath& path,
        const Vector& gridpoints,
        Vectors&, Vectors&, Vectors&,
        Matrices&, Vectors&)
    {
      throw std::logic_error("BoxConstraint should not be handled as standard LinearConstraint.");
    }

    virtual void computeBounds_impl (const GeometricPath& path, const Vector& gridpoint,
        Bounds& ubound, Bounds& xbound) = 0;

  private:
    bool hasXbounds_, hasUbounds_;
}; // class BoxConstraint

inline std::ostream& operator<< (std::ostream& os, const LinearConstraint& lc)
{
  return lc.print(os);
}

} // namespace toppra

#endif
