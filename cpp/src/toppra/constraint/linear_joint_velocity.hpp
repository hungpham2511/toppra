#ifndef TOPPRA_CONSTRAINT_LINEAR_JOINT_VELOCITY_HPP
#define TOPPRA_CONSTRAINT_LINEAR_JOINT_VELOCITY_HPP

#include <toppra/constraint.hpp>

namespace toppra {
namespace constraint {

/**
\brief A Joint Velocity Constraint class.

This class implements the case of constant velocity limits but can be derived to
achieve varying velocity limits. E.g.
\code
class LinearJointVelocityVarying : public LinearJointVelocity {
public:
  LinearJointVelocityVarying(...) : LinearJointVelocity (nDof) {}
protected:
  void computeVelocityLimits(value_type time)
  {
    m_lower = ...;
    m_upper = ...;
  }
};
\endcode
*/
class LinearJointVelocity : public LinearConstraint {
  public:
    LinearJointVelocity (const Vector& lowerVlimit, const Vector& upperVlimit)
      : LinearConstraint (0, 0, true, false, true)
      , m_lower (lowerVlimit)
      , m_upper (upperVlimit)
      , m_maxsd (1e8)
    {
      check();
    }

    /** Set the maximum allowed value of \f$\dot s\f$.
     * \param maxsd should be strictly positive.
     * */
    void maxSDot (value_type maxsd)
    {
      assert(maxsd > 0);
      m_maxsd = maxsd;
    }

    virtual std::ostream& print(std::ostream& os) const;

  protected:
    LinearJointVelocity (const int nDof)
      : LinearConstraint (0, 0, true, false, true)
      , m_lower (nDof)
      , m_upper (nDof)
      , m_maxsd (1e8)
    {
      check();
    }

    /**
      \brief Computes the velocity limit at time \c time.

      The result must be stored into attributes
      LinearJointVelocity::m_lower and LinearJointVelocity::m_upper.
      */
    virtual void computeVelocityLimits(value_type time) { (void)time; }

    /// The lower velocity limits
    Vector m_lower;
    /// The upper velocity limits
    Vector m_upper;

  private:
    void check();

    void computeParams_impl(const GeometricPath& path,
        const Vector& gridpoints,
        Vectors& a, Vectors& b, Vectors& c,
        Matrices& F, Vectors& g,
        Bounds& ubound, Bounds& xbound);

    value_type m_maxsd;
}; // class LinearJointVelocity
} // namespace constraint
} // namespace toppra

#endif
