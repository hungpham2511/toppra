#include <toppra/constraint/cartesian_velocity_norm.hpp>

#include <toppra/geometric_path.hpp>

namespace toppra {
namespace constraint {

std::ostream& CartesianVelocityNorm::print (std::ostream& os) const
{
  os << "CartesianVelocityNorm\n";
  return LinearConstraint::print(os) <<
    "    Velocity norm limit: " << m_limit << "\n";
}

void CartesianVelocityNorm::check ()
{
  if (m_limit < 0)
    throw std::invalid_argument("Velocity limit should be positive.");
  if (m_S.rows() != 6 || m_S.cols() != 6)
    throw std::invalid_argument("S matrix should be of size 6x6.");
}

void CartesianVelocityNorm::computeParams_impl(const GeometricPath& path,
        const Vector& gridpoints,
        Vectors& a, Vectors& b, Vectors& c,
        Matrices& F, Vectors& g,
        Bounds&, Bounds&)
{
  Eigen::Index N_1 = gridpoints.size();

  if (constantF()) {
    F[0].setIdentity();
    g[0][0] = m_limit;
  }

  Vector q(path.configSize()), qdot(path.dof()), v(6), S_v(6);
  for (std::size_t i = 0; i < N_1; ++i) {
    computeVelocityLimit(gridpoints[i]);

    q = path.eval_single (gridpoints[i], 0);
    qdot = path.eval_single (gridpoints[i], 1);
    computeVelocity(q, qdot, v);
    S_v = m_S * v;

    a[i].setZero();
    b[i].noalias() = v.transpose() * S_v;
    c[i].setZero();

    if (!constantF()) {
      F[i].setIdentity();
      g[i][0] = m_limit;
    }
  }
}

} // namespace constraint
} // namespace toppra
