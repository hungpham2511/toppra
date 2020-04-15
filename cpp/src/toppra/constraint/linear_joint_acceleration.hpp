#ifndef TOPPRA_CONSTRAINT_LINEAR_JOINT_ACCELERATION_HPP
#define TOPPRA_CONSTRAINT_LINEAR_JOINT_ACCELERATION_HPP

#include <toppra/constraint.hpp>

namespace toppra {
namespace constraint {

/// A Joint Acceleration Constraint class.
class LinearJointAcceleration : public LinearConstraint {
  public:
    LinearJointAcceleration (const Vector& lowerAlimit, const Vector& upperAlimit)
      : LinearConstraint (lowerAlimit.size() * 2, lowerAlimit.size())
      , lower_ (lowerAlimit)
      , upper_ (upperAlimit)
    {
      constantF_ = true;

      check();
    }

    virtual std::ostream& print(std::ostream& os) const;

  private:
    void check();

    void computeParams_impl(const GeometricPath& path,
        const Vector& gridpoints,
        Vectors& a, Vectors& b, Vectors& c,
        Matrices& F, Vectors& g);

    Vector lower_, upper_;
}; // class LinearJointAcceleration
} // namespace constraint
} // namespace toppra

#endif
