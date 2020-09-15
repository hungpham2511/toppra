#ifndef TOPPRA_CONSTRAINT_JOINT_TORQUE_HPP
#define TOPPRA_CONSTRAINT_JOINT_TORQUE_HPP

#include <toppra/constraint.hpp>

namespace toppra {
namespace constraint {

/** Base class for joint torque constraints.
 *
 * A joint torque constraint is given by
 * \f[
 *   A(q) \ddot q + \dot q^\top B(q) \dot q + C(q) + D( \dot q )= \tau
 * \f]
 *
 * where w is a vector that satisfies the polyhedral constraint:
 * \f[ F \tau \leq g \f]
 *
 * Notice that \f$ inverseDynamics(q, qd, qdd) = \tau\f$ and that
 * \f$ F, g \f$ are independant of the GeometricPath.
 *
 * To evaluate the constraint on a geometric path `p(s)`, multiple
 * calls to \ref computeInverseDynamics are made. Specifically one
 * can derive the second-order equation as follows
 *
 * \f{eqnarray}
 *   A(q) p'(s) \ddot s &+ [A(q) p''(s) + p'(s)^\top B(q) p'(s)] \dot s^2 &+ C(q) + D( \dot q ) &= \tau \\
 *         a(s) \ddot s &+                                  b(s) \dot s^2 &+ c(s) &= \tau
 * \f}
 *
 * */
class JointTorque : public LinearConstraint {
  public:
    virtual ~JointTorque () {}

    virtual std::ostream& print(std::ostream& os) const;

    /** Computes the joint torques from
     * \param q robot configuration
     * \param v robot velocity
     * \param a robot acceleration
     * \param[out] joint torques
     * */
    virtual void computeInverseDynamics (const Vector& q, const Vector& v, const Vector& a,
        Vector& tau) = 0;

    const Vector& lowerBounds () const
    { return m_lower; }

    void lowerBounds (const Vector& lb)
    {
      assert(lb.size() == m_lower.size());
      m_lower = lb;
    }

    const Vector& upperBounds () const
    { return m_upper; }

    void upperBounds (const Vector& ub)
    {
      assert(ub.size() == m_upper.size());
      m_upper = ub;
    }

    const Vector& frictionCoeffs () const
    { return m_frictionCoeffs; }

    void frictionCoeffs (const Vector& fc)
    {
      assert(fc.size() == m_frictionCoeffs.size());
      m_frictionCoeffs = fc;
    }

  protected:
    /**
     * \param lowerTlimit lower torque limit
     * \param upperTlimit upper torque limit
     * \param frictionCoeffs dry friction coefficients of each joint.
     *        size 0 when not considering friction.
     * */
    JointTorque (const Vector& lowerTlimit, const Vector& upperTlimit,
        const Vector& frictionCoeffs)
      : LinearConstraint (2*lowerTlimit.size(), lowerTlimit.size(), true, false, false)
      , m_lower (lowerTlimit)
      , m_upper (upperTlimit)
      , m_frictionCoeffs (frictionCoeffs)
    {
      check();
    }

    /// Move-assignment constructor
    JointTorque (JointTorque&& other)
      : LinearConstraint (other)
      , m_lower(std::move(other.m_lower))
      , m_upper(std::move(other.m_upper))
      , m_frictionCoeffs(std::move(other.m_frictionCoeffs))
    {}

  private:
    void check();

    void computeParams_impl(const GeometricPath& path,
        const Vector& gridpoints,
        Vectors& a, Vectors& b, Vectors& c,
        Matrices& F, Vectors& g,
        Bounds& ubound, Bounds& xbound);

    Vector m_lower, m_upper, m_frictionCoeffs;
}; // class JointTorque
} // namespace constraint
} // namespace toppra

#endif
