#ifndef TOPPRA_CONSTRAINT_LINEAR_JOINT_ACCELERATION_HPP
#define TOPPRA_CONSTRAINT_LINEAR_JOINT_ACCELERATION_HPP

#include <toppra/constraint.hpp>

namespace toppra {
namespace constraint {

/// A Joint Acceleration Constraint class.
class LinearJointAcceleration : public LinearConstraint {
  public:
    LinearJointAcceleration (const Vector& lowerAlimit, const Vector& upperAlimit)
      : LinearConstraint (lowerAlimit.size() * 2, lowerAlimit.size(), true, false, false)
      , m_lower (lowerAlimit)
      , m_upper (upperAlimit)
    {
      check();
    }

    virtual std::ostream& print(std::ostream& os) const;

  private:
    void check();

    void computeParams_impl(const GeometricPath& path,
        const Vector& gridpoints,
        Vectors& a, Vectors& b, Vectors& c,
        Matrices& F, Vectors& g,
        Bounds& ubound, Bounds& xbound);

    Vector m_lower, m_upper;
}; // class LinearJointAcceleration
} // namespace constraint
} // namespace toppra

#endif
