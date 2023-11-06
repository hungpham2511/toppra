#include <toppra/constraint/linear_joint_acceleration.hpp>

#include <toppra/geometric_path.hpp>

namespace toppra {
namespace constraint {

std::ostream& LinearJointAcceleration::print (std::ostream& os) const
{
  os << "LinearJointAcceleration\n";
  return LinearConstraint::print(os) <<
    "    Lower acceleration limit: " << m_lower.transpose() << "\n"
    "    Upper acceleration limit: " << m_upper.transpose() << "\n";
}

void LinearJointAcceleration::check ()
{
  if ((m_lower.array() > m_upper.array()).any())
    throw std::invalid_argument("Bad acceleration limits.");
}

void LinearJointAcceleration::computeParams_impl(const GeometricPath& path,
        const Vector& times,
        Vectors& a, Vectors& b, Vectors& c,
        Matrices& F, Vectors& g,
        Bounds&, Bounds&)
{
  Eigen::Index N_1 = times.size();

  Eigen::Index ndofs (nbVariables());

  if (constantF()) {
    // Compute F and g
    Matrix& _F = F[0];
    _F.topRows(ndofs).setIdentity();
    _F.bottomRows(ndofs).setZero();
    _F.bottomRows(ndofs).diagonal().setConstant(-1);
    Vector& _g = g[0];
    _g.head(ndofs) =  m_upper;
    _g.tail(ndofs) = -m_lower;
  }

  assert(ndofs == path.dof());
  for (std::size_t i = 0; i < N_1; ++i) {
    computeAccelerationLimits(times[i]);

    a[i] = path.eval_single(times[i], 1);
    assert(a[i].size() == ndofs);
    b[i] = path.eval_single(times[i], 2);
    assert(b[i].size() == ndofs);
    c[i].setZero();

    if (!constantF()) {
      Matrix& _F = F[i];
      _F.topRows(ndofs).setIdentity();
      _F.bottomRows(ndofs).setZero();
      _F.bottomRows(ndofs).diagonal().setConstant(-1);
      Vector& _g = g[i];
      _g.head(ndofs) =  m_upper;
      _g.tail(ndofs) = -m_lower;
    }
  }
}

} // namespace constraint
} // namespace toppra
