#include <toppra/constraint.hpp>

#include <toppra/geometric_path.hpp>

namespace toppra {

std::ostream& LinearConstraint::print(std::ostream& os) const
{
  return os <<
    "    Discretization Scheme: " << m_discretizationType << "\n"
    "    Has " << (hasLinearInequalities() ? "linear inequalities, " : "")
    << (hasUbounds() ? "bounds on u, " : "")
    << (hasXbounds() ? "bounds on x, " : "") << "\n";
}

void LinearConstraint::discretizationType (DiscretizationType type)
{
  // I don't think the check done in Python is useful in C++.
  m_discretizationType = type;
}

/// \cond INTERNAL

/// \param k number of constraints
/// \param m number of variables
void allocateLinearPart(std::size_t N, Eigen::Index k, Eigen::Index m,
    bool constantF, Vectors& a, Vectors& b, Vectors& c, Matrices& F, Vectors& g)
  TOPPRA_NO_EXPORT;

void allocateLinearPart(std::size_t N, Eigen::Index k, Eigen::Index m,
    bool constantF, Vectors& a, Vectors& b, Vectors& c, Matrices& F, Vectors& g)
{
  a.resize(N);
  b.resize(N);
  c.resize(N);
  if (constantF) {
    F.resize(1);
    g.resize(1);
  } else {
    F.resize(N);
    g.resize(N);
  }
  for (Vector& x : a) x.resize(m);
  for (Vector& x : b) x.resize(m);
  for (Vector& x : c) x.resize(m);
  for (Matrix& x : F) x.resize(k, m);
  for (Vector& x : g) x.resize(k);
}

void checkSizes (std::size_t N, Eigen::Index k, Eigen::Index m,
    bool constantF, Vectors& a, Vectors& b, Vectors& c, Matrices& F, Vectors& g)
  TOPPRA_NO_EXPORT;

void checkSizes (std::size_t N, Eigen::Index k, Eigen::Index m,
    bool constantF, Vectors& a, Vectors& b, Vectors& c, Matrices& F, Vectors& g)
{
  if (a.size() != N)
    throw std::invalid_argument("Wrong number of a vectors");
  if (b.size() != N)
    throw std::invalid_argument("Wrong number of b vectors");
  if (c.size() != N)
    throw std::invalid_argument("Wrong number of c vectors");
  if (constantF && F.size() != 1)
    throw std::invalid_argument("Expected only one F matrix");
  if (constantF && g.size() != 1)
    throw std::invalid_argument("Expected only one g matrix");
  if (!constantF && g.size() != N)
    throw std::invalid_argument("Wrong number of g matrices");

  for (std::size_t i = 0; i < N; ++i) {
    if (a[i].size() != m)
      throw std::invalid_argument("Wrong a[i] vector size.");
    if (b[i].size() != m)
      throw std::invalid_argument("Wrong b[i] vector size.");
    if (c[i].size() != m)
      throw std::invalid_argument("Wrong c[i] vector size.");
    if (constantF) {
      if (i == 0 && (F[0].rows() != k || F[0].cols() != m))
        throw std::invalid_argument("Wrong F[0] matrix dimensions.");
      if (i == 0 && g[0].size() != k)
        throw std::invalid_argument("Wrong g[0] vector size.");
    } else {
      if (F[i].rows() != k || F[i].cols() != m)
        throw std::invalid_argument("Wrong F[i] matrix dimensions.");
      if (g[i].size() != k)
        throw std::invalid_argument("Wrong g[i] vector size.");
    }
  }
}

/// Convert from Collocation to Interpolation
void collocationToInterpolate (const Vector& gridpoints,
    bool constantF,
    const Vectors& a_col, const Vectors& b_col, const Vectors& c_col,
    const Matrices& F_col, const Vectors& g_col,
    Vectors& a_intp, Vectors& b_intp, Vectors& c_intp,
    Matrices& F_intp, Vectors& g_intp)
  TOPPRA_NO_EXPORT;

void collocationToInterpolate (const Vector& gridpoints,
    bool constantF,
    const Vectors& a_col, const Vectors& b_col, const Vectors& c_col,
    const Matrices& F_col, const Vectors& g_col,
    Vectors& a_intp, Vectors& b_intp, Vectors& c_intp,
    Matrices& F_intp, Vectors& g_intp)
{
  std::size_t N (gridpoints.size()-1);
  Vector deltas (gridpoints.tail(N) - gridpoints.head(N));
  Eigen::Index m (a_col[0].size()),
               k (g_col[0].size());

  //a_intp[:, :d] = a
  //a_intp[:-1, d:] = a[1:] + 2 * deltas.reshape(-1, 1) * b[1:]
  //a_intp[-1, d:] = a_intp[-1, :d]
  for (std::size_t i = 0; i <= N; ++i) {
    a_intp[i].head(m) = a_col[i];
    if (i < N)
      a_intp[i].tail(m) = a_col[i+1] + 2 * deltas[i] * b_col[i+1];
    else
      a_intp[N].tail(m) = a_col[N];
  }

  // b_intp[:, :d] = b
  // b_intp[:-1, d:] = b[1:]
  // b_intp[-1, d:] = b_intp[-1, :d]
  for (std::size_t i = 0; i <= N; ++i) {
    b_intp[i].head(m) = b_col[i];
    b_intp[i].tail(m) = b_col[std::min(i+1, N)];
  }

  // c_intp[:, :d] = c
  // c_intp[:-1, d:] = c[1:]
  // c_intp[-1, d:] = c_intp[-1, :d]
  for (std::size_t i = 0; i <= N; ++i) {
    c_intp[i].head(m) = c_col[i];
    c_intp[i].tail(m) = c_col[std::min(i+1, N)];
  }

  const auto zero (Matrix::Zero (k, m));
  if (constantF) {
    g_intp[0] << g_col[0], g_col[0];

    F_intp[0] << F_col[0], zero,
                 zero, F_col[0];
  } else {
    // g_intp[:, :m] = g
    // g_intp[:-1, m:] = g[1:]
    // g_intp[-1, m:] = g_intp[-1, :m]
    for (std::size_t i = 0; i <= N; ++i) {
      g_intp[i].head(k) = g_col[i];
      g_intp[i].tail(k) = g_col[std::min(i+1,N)];
    }

    // F_intp[:, :m, :d] = F
    // F_intp[:-1, m:, d:] = F[1:]
    // F_intp[-1, m:, d:] = F[-1]
    for (std::size_t i = 0; i <= N; ++i) {
      F_intp[i] << F_col[i], zero,
                   zero, F_col[std::min(i+1,N)];
    }
  }
}

/// \endcond

void LinearConstraint::allocateParams(std::size_t N,
    Vectors& a, Vectors& b, Vectors& c, Matrices& F, Vectors& g,
    Bounds ubound, Bounds& xbound)
{
  if (hasLinearInequalities()) {
    Eigen::Index m (nbVariables()), k (nbConstraints());
    if (m_discretizationType == Interpolation) {
      m *= 2;
      k *= 2;
    }

    allocateLinearPart (N, k, m, constantF(), a, b, c, F, g);
  }
  if (hasUbounds())
    ubound.resize(N);
  if (hasXbounds())
    xbound.resize(N);
}

void LinearConstraint::computeParams(const GeometricPath& path, const Vector& gridpoints,
    Vectors& a, Vectors& b, Vectors& c, Matrices& F, Vectors& g,
    Bounds& ubound, Bounds& xbound)
{
  Eigen::Index N = gridpoints.size();
  assert (N > 0);
  allocateParams(gridpoints.size(), a, b, c, F, g, ubound, xbound);

  if (m_discretizationType == Interpolation && hasLinearInequalities()) {
    Vectors a_col, b_col, c_col, g_col;
    Matrices F_col;
    allocateLinearPart (N, m_k, m_m, constantF(), a_col, b_col, c_col, F_col, g_col);
    computeParams_impl(path, gridpoints, a_col, b_col, c_col, F_col, g_col, ubound, xbound);
    collocationToInterpolate(gridpoints, constantF(),
        a_col, b_col, c_col, F_col, g_col,
        a, b, c, F, g);
  } else {
    computeParams_impl(path, gridpoints, a, b, c, F, g, ubound, xbound);
  }
}

} // namespace toppra
