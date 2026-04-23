#include <toppra/constraint/linear_joint_velocity.hpp>

#include <toppra/geometric_path.hpp>

namespace toppra {
namespace constraint {

std::ostream& LinearJointVelocity::print (std::ostream& os) const
{
  os << "LinearJointVelocity\n";
  return LinearConstraint::print(os) <<
    "    Lower velocity limit: " << m_lower.transpose() << "\n"
    "    Upper velocity limit: " << m_upper.transpose() << "\n";
}

void LinearJointVelocity::check ()
{
  if (m_lower.size() != m_upper.size())
    throw std::invalid_argument("Velocity limits size must match.");
  if ((m_lower.array() > m_upper.array()).any())
    throw std::invalid_argument("Bad velocity limits.");
}

void LinearJointVelocity::computeParams_impl(const GeometricPath& path,
        const Vector& gridpoints,
        Vectors&, Vectors&, Vectors&, Matrices&, Vectors&,
        Bounds&, Bounds& xbound)
{
  Eigen::Index N_1 = gridpoints.size();
  Eigen::Index ndofs (m_lower.size());
  assert(path.dof() == m_lower.size());

  Vector v, v_inv(ndofs), lb_v(ndofs);
  for (std::size_t i = 0; i < N_1; ++i) {
    computeVelocityLimits(gridpoints[i]);

    v = path.eval_single (gridpoints[i], 1);
    assert(ndofs == v.size());

    v_inv.noalias() = v.cwiseInverse();

    value_type sdmin = - m_maxsd,
               sdmax =   m_maxsd;
    for (Eigen::Index k = 0; k < ndofs; ++k) {
      if (v[k] > 0) {
        sdmax = std::min(m_upper[k] * v_inv[k], sdmax);
        sdmin = std::max(m_lower[k] * v_inv[k], sdmin);
      } else if (v[k] < 0) {
        sdmax = std::min(m_lower[k] * v_inv[k], sdmax);
        sdmin = std::max(m_upper[k] * v_inv[k], sdmin);
      } else {
        if (m_upper[k] < 0 || m_lower[k] > 0)
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
