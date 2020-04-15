#include <toppra/constraint.hpp>

namespace toppra {

std::ostream& LinearConstraint::print(std::ostream& os) const
{
  os << "LinearConstraint\n"
        "    Type: " << constraintType_ << "\n"
        "    Discretization Scheme: " << discretizationType_ << "\n";
  return os;
}

void LinearConstraint::discretizationType (DiscretizationType type)
{
  // I don't think the check done in Python is useful in C++.
  discretizationType_ = type;
}

void LinearConstraint::computeParams(const GeometricPath& path, const Vector& gridpoints,
    Vectors& a, Vectors& b, Vectors& c, Matrices& F, Vectors& g, Bounds& ubound, Bounds& xbound)
{
  Eigen::Index N_1 = gridpoints.size();
  assert (N_1 > 0);
  if (a.size() != N_1)
    std::invalid_argument("Wrong number of a vectors");
  if (b.size() != N_1)
    std::invalid_argument("Wrong number of b vectors");
  if (c.size() != N_1)
    std::invalid_argument("Wrong number of c vectors");
  if (F.size() != N_1)
    std::invalid_argument("Wrong number of F matrices");
  if (g.size() != N_1)
    std::invalid_argument("Wrong number of g vectors");

  Eigen::Index m = a[0].cols(),
               k = g[0].cols();
  for (std::size_t i = 0; i < N_1; ++i) {
    if (a[i].size() != m)
      std::invalid_argument("Wrong a[i] vector size.");
    if (b[i].size() != m)
      std::invalid_argument("Wrong b[i] vector size.");
    if (c[i].size() != m)
      std::invalid_argument("Wrong c[i] vector size.");
    if (F[i].rows() != k || F[i].cols() != m)
      std::invalid_argument("Wrong F[i] matrix dimensions.");
    if (g[i].size() != k)
      std::invalid_argument("Wrong g[i] vector size.");
  }

  computeParams_impl(path, gridpoints, a, b, c, F, g, ubound, xbound);
}

} // namespace toppra
