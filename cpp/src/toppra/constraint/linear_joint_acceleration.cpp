
#include <toppra/constraint/linear_joint_acceleration.hpp>

namespace toppra {
namespace constraint {

std::ostream& LinearJointAcceleration::print (std::ostream& os) const
{
  os << "LinearJointAcceleration\n";
  return LinearConstraint::print(os) <<
    "    Lower acceleration limit: " << lower_.transpose() << "\n"
    "    Upper acceleration limit: " << upper_.transpose() << "\n";
}

void LinearJointAcceleration::check ()
{
  if ((lower_.array() > upper_.array()).any())
    throw std::invalid_argument("Bad acceleration limits.");
}

void LinearJointAcceleration::computeParams_impl(const GeometricPath& path,
        const Vector& gridpoints,
        Vectors& a, Vectors& b, Vectors& c,
        Matrices& F, Vectors& g,
        Bounds, Bounds&)
{
  Eigen::Index N_1 = gridpoints.size();

  Eigen::Index ndofs (nbVariables());

  // Compute F and g
  Matrix& _F = F[0];
  _F.   topRows(ndofs).setIdentity();
  _F.bottomRows(ndofs).setZero();
  _F.bottomRows(ndofs).diagonal().setConstant(-1);
  Vector& _g = g[0];
  _g.head(ndofs) =  upper_;
  _g.tail(ndofs) = -lower_;

  /// \todo Use GeometricPath evaluation
  // assert(ndofs == vs[0].size());
  // a <- (path(gridpoints / scaling, order=1) / scaling).reshape((-1, path.dof))
  // b <- (path(gridpoints / scaling, order=2) / scaling ** 2).reshape((-1, path.dof))
  for (std::size_t i = 0; i < N_1; ++i) c[i].setZero();
}

} // namespace constraint
} // namespace toppra
