#ifndef TOPPRA_CONSTRAINT_LINEAR_JOINT_VELOCITY_HPP
#define TOPPRA_CONSTRAINT_LINEAR_JOINT_VELOCITY_HPP

#include <toppra/constraint.hpp>

namespace toppra {
namespace constraint {

/// A Joint Velocity Constraint class.
class LinearJointVelocity : public BoxConstraint {
  public:
    LinearJointVelocity (const Vector& lowerVlimit, const Vector& upperVlimit)
      : BoxConstraint (false, true)
      , lower_ (lowerVlimit)
      , upper_ (upperVlimit)
      , maxsd_ (1e8)
    {
      check();
    }

    /** Set the maximum allowed value of \f$\dot s\f$.
     * \param maxsd should be strictly positive.
     * */
    void maxSDot (value_type maxsd)
    {
      assert(maxsd > 0);
      maxsd_ = maxsd;
    }

    virtual std::ostream& print(std::ostream& os) const;

  private:
    void check();

    void computeBounds_impl (const GeometricPath& path, const Vector& gridpoint,
        Bounds& ubound, Bounds& xbound);

    Vector lower_, upper_;
    value_type maxsd_;
}; // class LinearJointVelocity
} // namespace constraint
} // namespace toppra

#endif
