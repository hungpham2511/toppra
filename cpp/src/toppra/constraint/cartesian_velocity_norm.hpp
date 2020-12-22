#ifndef TOPPRA_CONSTRAINT_CARTESIAN_VELOCITY_NORM_HPP
#define TOPPRA_CONSTRAINT_CARTESIAN_VELOCITY_NORM_HPP

#include <toppra/constraint.hpp>

namespace toppra {
namespace constraint {

/**
\brief A cartesian velocity constraint class.

Given a path \f$ p(s) \f$, this class constraints the norm
\f$||v||_S^2 = v^T S v \leq limit \f$ of the velocity \f$v\f$ of a frame.
As \f$ v = J(p(s)) p'(s) \dot{s} \f$, it fits into a LinearConstraint as follows.

\f{eqnarray}
    p'(s)^T J(q)^T S J(q) p'(s) \dot{s}^2 &= w \\
    w & \leq \mathbf g_i         \\
\f}

This class implements the case of constant velocity limits but can be derived to
achieve varying velocity limits. E.g.
\code
class CartesianVelocityVarying : public CartesianVelocityNorm {
public:
  CartesianVelocityVarying(...) : CartesianVelocityNorm () {}
protected:
  void computeVelocityLimit(value_type time)
  {
    // Eventually, if one of the two is constant, it can be set in the constructor.
    m_limit = ...;
    m_S = ...;
  }
};
\endcode
*/
class CartesianVelocityNorm : public LinearConstraint {
  public:
    virtual std::ostream& print(std::ostream& os) const;

  protected:
    /// Constructor for constant velocity limit.
    CartesianVelocityNorm (const Matrix& S, const double& limit)
      : LinearConstraint (1, 1, true, false, false)
      , m_limit (limit)
      , m_S (S)
    {
      check();
    }

    /// Constructor for varying velocity limit.
    /// \note Attributes \ref m_S and \ref m_limit **must** be computed in
    /// \ref computeVelocityLimit
    CartesianVelocityNorm ()
      : LinearConstraint (1, 1, false, false, false)
      , m_limit (1.)
      , m_S (6,6)
    {
      check();
    }

    /// Pure abstract method to compute the velocity from
    /// \param q the current configuration
    /// \param qdot the current velocity
    /// \retval v the 6D frame velocity
    virtual void computeVelocity (const Vector& q, const Vector& qdot,
        Vector& v) = 0;

    /**
      \brief Computes the velocity limit at time \c time.

      The result must be stored into attribute
      CartesianVelocityNorm::m_limit
      */
    virtual void computeVelocityLimit(value_type time) { (void)time; }

    /// The velocity limit
    value_type m_limit;
    /// The selection matrix
    Matrix m_S;

  private:
    void check();

    void computeParams_impl(const GeometricPath& path,
        const Vector& gridpoints,
        Vectors& a, Vectors& b, Vectors& c,
        Matrices& F, Vectors& g,
        Bounds& ubound, Bounds& xbound);
}; // class CartesianVelocityNorm
} // namespace constraint
} // namespace toppra

#endif
