#include <toppra/constraint/joint_torque.hpp>

#include <toppra/geometric_path.hpp>

namespace toppra {
namespace constraint {

std::ostream& JointTorque::print (std::ostream& os) const
{
  os << "JointTorque\n";
  return LinearConstraint::print(os) <<
    "    Lower torque limit: " << lower_.transpose() << "\n"
    "    Upper torque limit: " << upper_.transpose() << "\n";
}

void JointTorque::check ()
{
  if (lower_.size() != upper_.size())
    throw std::invalid_argument("Torque limits size must match.");
  if (lower_.size() != frictionCoeffs_.size())
    throw std::invalid_argument("Torque limits size and friction vector size must match.");
  if ((lower_.array() > upper_.array()).any())
    throw std::invalid_argument("Bad torque limits.");
}

void JointTorque::computeParams_impl(const GeometricPath& path,
        const Vector& times,
        Vectors& a, Vectors& b, Vectors& c, Matrices& F, Vectors& g,
        Bounds , Bounds&)
{
  Eigen::Index N = times.size();
  Eigen::Index ndofs (lower_.size());

  // Compute static F and g
  F[0].topRows(ndofs).setIdentity();
  F[0].bottomRows(ndofs).setZero();
  F[0].bottomRows(ndofs).diagonal().setConstant(-1);
  g[0].head(ndofs) =   upper_;
  g[0].tail(ndofs) = - lower_;

  Vector zero = Vector::Zero(ndofs);
  for (std::size_t i = 0; i < N; ++i) {
    Vector cfg = path.eval_single(times[i], 0);
    Vector vel = path.eval_single(times[i], 1);
    Vector acc = path.eval_single(times[i], 2);
    assert(vel.size() == ndofs);
    assert(acc.size() == ndofs);

    computeInverseDynamics(cfg, zero, zero, c[i]);
    computeInverseDynamics(cfg, zero, vel, a[i]);
    a[i] -=  c[i];
    computeInverseDynamics(cfg, vel, acc, b[i]);
    b[i] -=  c[i];

    c[i].array() += vel.array().sign() * frictionCoeffs_.array();
  }
}

} // namespace constraint
} // namespace toppra
