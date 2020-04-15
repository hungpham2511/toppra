#include <toppra/constraint/linear_joint_velocity.hpp>

namespace toppra {
namespace constraint {

std::ostream& LinearJointVelocity::print (std::ostream& os) const
{
  os << "LinearJointVelocity\n";
  return BoxConstraint::print(os) <<
    "    Lower velocity limit: " << lower_.transpose() << "\n"
    "    Upper velocity limit: " << upper_.transpose() << "\n";
}

void LinearJointVelocity::check ()
{
  if ((lower_.array() > upper_.array()).any())
    throw std::invalid_argument("Bad velocity limits.");
}

void LinearJointVelocity::computeBounds_impl (
    const GeometricPath& path, const Vector& gridpoint,
    Bounds&, Bounds& xbound)
{
  Eigen::Index N_1 = gridpoint.size();
  Eigen::Index ndofs (nbVariables());

  /// \todo Use GeometricPath evaluation
  Vectors vs(N_1, Vector(ndofs)); // (path(gridpoints / scaling, 1) / scaling
  assert(ndofs == vs[0].size());

  Vector v_inv(ndofs), lb_v(ndofs);
  for (std::size_t i = 0; i < N_1; ++i) {
    const Vector& v (vs[i]);
    v_inv.noalias() = v.cwiseInverse();

    value_type sdmin = - maxsd_,
               sdmax =   maxsd_;
    for (Eigen::Index k = 0; k < ndofs; ++k) {
      if (v[k] > 0) {
        sdmax = std::min(upper_[k] * v_inv[k], sdmax);
        sdmin = std::max(lower_[k] * v_inv[k], sdmin);
      } else if (v[k] < 0) {
        sdmax = std::min(lower_[k] * v_inv[k], sdmax);
        sdmin = std::max(upper_[k] * v_inv[k], sdmin);
      } else {
        if (upper_[k] < 0 || lower_[k] > 0)
          /// \todo the problem is infeasible. How should we inform the user ?
          throw std::runtime_error("BoxConstraint is infeasible");
      }
    }
    xbound[i][0] = 0;
    if (sdmin > 0) xbound[i][0] = sdmin*sdmin;
    xbound[i][1] = sdmax * sdmax;
  }
}

} // namespace constraint
} // namespace toppra
