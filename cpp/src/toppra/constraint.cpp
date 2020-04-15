#include <toppra/constraint.hpp>

namespace toppra {

std::ostream& LinearConstraint::print(std::ostream& os) const
{
  os << "LinearConstraint\n"
        "    Discretization Scheme: " << discretizationType_ << "\n";
  return os;
}

void LinearConstraint::discretizationType (DiscretizationType type)
{
  // I don't think the check done in Python is useful in C++.
  discretizationType_ = type;
}

void LinearConstraint::computeParams(const GeometricPath& path, const Vector& gridpoints,
    Vectors& a, Vectors& b, Vectors& c, Matrices& F, Vectors& g)
{
  Eigen::Index N_1 = gridpoints.size();
  assert (N_1 > 0);
  if (a.size() != N_1)
    throw std::invalid_argument("Wrong number of a vectors");
  if (b.size() != N_1)
    throw std::invalid_argument("Wrong number of b vectors");
  if (c.size() != N_1)
    throw std::invalid_argument("Wrong number of c vectors");
  if (constantF() && F.size() != 1)
    throw std::invalid_argument("Expected only one F matrix");
  if (!constantF() && F.size() != N_1)
    throw std::invalid_argument("Wrong number of F matrices");
  if (g.size() != N_1)
    throw std::invalid_argument("Wrong number of g vectors");

  for (std::size_t i = 0; i < N_1; ++i) {
    if (a[i].size() != m_)
      throw std::invalid_argument("Wrong a[i] vector size.");
    if (b[i].size() != m_)
      throw std::invalid_argument("Wrong b[i] vector size.");
    if (c[i].size() != m_)
      throw std::invalid_argument("Wrong c[i] vector size.");
    if (constantF())
      if (i == 0 && (F[0].rows() != k_ || F[0].cols() != m_))
        throw std::invalid_argument("Wrong F[0] matrix dimensions.");
    else
      if (F[i].rows() != k_ || F[i].cols() != m_)
        throw std::invalid_argument("Wrong F[i] matrix dimensions.");
    if (g[i].size() != k_)
      throw std::invalid_argument("Wrong g[i] vector size.");
  }

  computeParams_impl(path, gridpoints, a, b, c, F, g);
}

std::ostream& BoxConstraint::print(std::ostream& os) const
{
  os << "BoxConstraint on " <<
    ((hasUbounds_ && hasXbounds_) ? "u and v" : (hasUbounds_ ? "u" : "v")) << "\n"
    "    Discretization Scheme: " << discretizationType_ << "\n";
  return os;
}

void BoxConstraint::computeBounds(const GeometricPath& path, const Vector& gridpoints,
    Bounds& ubound, Bounds& xbound)
{
  Eigen::Index N_1 = gridpoints.size();
  assert (N_1 > 0);
  if (hasUbounds() && ubound.size() != N_1)
    throw std::invalid_argument("Wrong ubound vector size.");
  if (hasXbounds() && xbound.size() != N_1)
    throw std::invalid_argument("Wrong ubound vector size.");

  computeBounds_impl(path, gridpoints, ubound, xbound);
}

} // namespace toppra
