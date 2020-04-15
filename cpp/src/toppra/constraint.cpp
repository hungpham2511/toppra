#include <toppra/constraint.hpp>

#include <toppra/geometric_path.hpp>

namespace toppra {

std::ostream& LinearConstraint::print(std::ostream& os) const
{
  return os << "    Discretization Scheme: " << discretizationType_ << "\n";
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
    if (constantF()) {
      if (i == 0 && (F[0].rows() != k_ || F[0].cols() != m_))
        throw std::invalid_argument("Wrong F[0] matrix dimensions.");
      if (i == 0 && g[0].size() != k_)
        throw std::invalid_argument("Wrong g[0] vector size.");
    } else {
      if (F[i].rows() != k_ || F[i].cols() != m_)
        throw std::invalid_argument("Wrong F[i] matrix dimensions.");
      if (g[i].size() != k_)
        throw std::invalid_argument("Wrong g[i] vector size.");
    }
  }

  computeParams_impl(path, gridpoints, a, b, c, F, g);
  if (discretizationType_ == Interpolation)
    collocationToInterpolate(gridpoints, a, b, c, F, g);
}

void LinearConstraint::collocationToInterpolate (const Vector& gridpoints,
    Vectors& a, Vectors& b, Vectors& c,
    Matrices& F, Vectors& g)
{
  std::size_t N (gridpoints.size()-1);
  Vector deltas (gridpoints.tail(N) - gridpoints.head(N));

  Vectors a_intp (N+1);
  //a_intp[:, :d] = a
  //a_intp[:-1, d:] = a[1:] + 2 * deltas.reshape(-1, 1) * b[1:]
  //a_intp[-1, d:] = a_intp[-1, :d]
  for (std::size_t i = 0; i <= N; ++i) {
    a_intp[i].resize(2*m_);
    a_intp[i].head(m_) = a[i];
    if (i < N)
      a_intp[i].tail(m_) = a[i+1] + 2 * deltas.cwiseProduct(b[i+1].tail(N));
    else
      a_intp[N].tail(m_) = a[N];
  }

  Vectors b_intp (N+1);
  // b_intp[:, :d] = b
  // b_intp[:-1, d:] = b[1:]
  // b_intp[-1, d:] = b_intp[-1, :d]
  for (std::size_t i = 0; i <= N; ++i) {
    b_intp[i].resize(2*m_);
    b_intp[i].head(m_) = b[i];
    b_intp[i].tail(m_) = b[std::min(i+1, N)];
  }

  Vectors c_intp (N+1);
  // c_intp[:, :d] = c
  // c_intp[:-1, d:] = c[1:]
  // c_intp[-1, d:] = c_intp[-1, :d]
  for (std::size_t i = 0; i <= N; ++i) {
    c_intp[i].resize(2*m_);
    c_intp[i].head(m_) = c[i];
    c_intp[i].tail(m_) = c[std::min(i+1, N)];
  }

  const auto zero (Matrix::Zero (2 * k_, 2 * m_));
  Vectors g_intp;
  Matrices F_intp;
  if (constantF()) {
    g_intp.resize (1);
    g_intp[0].resize(2 * k_);
    g_intp[0] << g[0], g[0];

    F_intp.resize (1);
    F_intp[0].resize(2 * k_, 2 * m_);
    F_intp[0] << F[0], zero,
                 zero, F[0];
  } else {
    g_intp.resize (N+1);
    // g_intp[:, :m] = g
    // g_intp[:-1, m:] = g[1:]
    // g_intp[-1, m:] = g_intp[-1, :m]
    for (std::size_t i = 0; i <= N; ++i) {
      g_intp[i].resize(2 * k_);
      g_intp[i].head(k_) = g[i];
      g_intp[i].tail(k_) = g[std::min(i+1,N)];
    }

    F_intp.resize (N+1);
    // F_intp[:, :m, :d] = F
    // F_intp[:-1, m:, d:] = F[1:]
    // F_intp[-1, m:, d:] = F[-1]
    for (std::size_t i = 0; i <= N; ++i) {
      F_intp[i].resize(2 * k_, 2 * m_);
      F_intp[i] << F[i], zero,
                   zero, F[std::min(i+1,N)];
    }
  }
  a.swap(a_intp);
  b.swap(b_intp);
  c.swap(c_intp);
  F.swap(F_intp);
  g.swap(g_intp);
}

std::ostream& BoxConstraint::print(std::ostream& os) const
{
  os << "    act on " << ((hasUbounds_ && hasXbounds_)
      ? "u and v"
      : (hasUbounds_ ? "u" : "v")) << "\n";
  return LinearConstraint::print(os);
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
