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
    Collocation,   ///< smaller problem size, but lower accuracy.
    Interpolation, ///< larger problem size, but higher accuracy.
};

/** \brief Abstract interface for the constraints.
 *
 *  Also known as Second-order Constraint.
 *
 *  A Canonical Linear Constraint has the following form:
 *  \f{eqnarray}
 *      \mathbf a_i u + \mathbf b_i x + \mathbf c_i &= v \\
 *      \mathbf F_i v & \leq \mathbf g_i                \\
 *      x^b_{i, 0} \leq x & \leq x^b_{i, 1} \\
 *      u^b_{i, 0} \leq u & \leq u^b_{i, 1}
 *  \f}
 *
 *  Alternatively, if \f$ \mathbf F_i \f$ is constant for all values
 *  of \f$i\f$, then we can consider the simpler constraint:
 *  \f[
 *      \mathbf{F} v \leq \mathbf g
 *  \f]
 *
 *  In this case, the returned value of \f$F\f$ by
 *  LinearConstraint::computeParams has shape (k, m) instead of (N, k, m),
 *  \f$ g \f$ (k) instead of (N, k) and the class attribute
 *  LinearConstraint::constantF will be \c true.
 *
 *  \note Derived classes should at least implement the method
 *  LinearConstraint::computeParams_impl.
 *
 *  \sa JointAccelerationConstraint, JointVelocityConstraint,
 *  CanonicalLinearSecondOrderConstraint
 *
 * */
class LinearConstraint {
  public:
    DiscretizationType discretizationType () const
    {
      return m_discretizationType;
    }

    void discretizationType (DiscretizationType type);

    /** Tells whether \f$ F, g \f$ matrices are the same over all the grid points.
     * In this case, LinearConstraint::computeParams F and g parameters should
     * only be of size 1.
     * */
    bool constantF () const
    {
      return m_constantF;
    }

    /// Dimension of \f$g\f$.
    Eigen::Index nbConstraints () const
    {
      return m_k;
    }

    /// Dimension of \f$a, b, c, v\f$.
    Eigen::Index nbVariables () const
    {
      return m_m;
    }

    bool hasLinearInequalities () const
    {
      return nbConstraints() > 0;
    }

    /** Whether this constraint has bounds on \f$u\f$.
     * */
    bool hasUbounds () const
    {
      return m_hasUbounds;
    }

    /** Whether this constraint has bounds on \f$x\f$.
     * */
    bool hasXbounds () const
    {
      return m_hasXbounds;
    }

    /**
     * \param N number of gripoints (i.e. the number of intervals + 1)
     * */
    void allocateParams (std::size_t N,
        Vectors& a, Vectors& b, Vectors& c,
        Matrices& F, Vectors& g,
        Bounds ubound, Bounds& xbound);

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
        Matrices& F, Vectors& g,
        Bounds& ubound, Bounds& xbound);

    virtual std::ostream& print(std::ostream& os) const;

    virtual ~LinearConstraint () {}

  protected:
    /**
     * \param k number of inequality constraints.
     * \param m number of internal variable (i.e. dimention of \f$v\f$).
     * \param constantF whether \f$F\f$ and \f$g\f$ are constant.
     * \param uBound whether \f$u\f$ is bounded.
     * \param xBound whether \f$x\f$ is bounded.
     * */
    LinearConstraint(Eigen::Index k, Eigen::Index m, bool constantF,
        bool uBound, bool xBound)
      : m_discretizationType (Collocation)
      , m_k (k), m_m (m)
      , m_constantF (constantF)
      , m_hasUbounds (uBound)
      , m_hasXbounds (xBound)
    {}

    virtual void computeParams_impl(const GeometricPath& path,
        const Vector& gridpoints,
        Vectors& a, Vectors& b, Vectors& c,
        Matrices& F, Vectors& g,
        Bounds& ubound, Bounds& xbound) = 0;

    Eigen::Index m_k, m_m;
    DiscretizationType m_discretizationType;
    bool m_constantF, m_hasUbounds, m_hasXbounds;
}; // class LinearConstraint

/// \brief write a LinearConstraint to an output stream
inline std::ostream& operator<< (std::ostream& os, const LinearConstraint& lc)
{
  return lc.print(os);
}

} // namespace toppra

#endif
