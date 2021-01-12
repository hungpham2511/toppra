#include <toppra/solver.hpp>

#include <toppra/constraint.hpp>
#include <toppra/geometric_path.hpp>

#ifdef BUILD_WITH_qpOASES
#include <toppra/solver/qpOASES-wrapper.hpp>
#endif
#ifdef BUILD_WITH_GLPK
#include <toppra/solver/glpk-wrapper.hpp>
#endif
#include <toppra/solver/seidel.hpp>

namespace toppra {

SolverPtr Solver::createDefault() {
#ifdef BUILD_WITH_qpOASES
    return std::make_shared<solver::qpOASESWrapper>();
#elif BUILD_WITH_GLPK
    return std::make_shared<solver::GLPKWrapper>();
#else
    return std::make_shared<solver::Seidel>();
#endif
}

void Solver::initialize (const LinearConstraintPtrs& constraints, const GeometricPathPtr& path,
    const Vector& times)
{
  m_constraints = constraints;
  m_path = path;
  m_times = times;
  m_N = times.size()-1;
  m_nV = 2;
  m_deltas = times.tail(m_N) - times.head(m_N);

  if ((m_deltas.array() <= 0).any())
    throw std::invalid_argument("Invalid times.");
  /// \todo assert that the time range of the path equals [ times[0], times[m_N] ].

  // Compute the constraints parameters
  LinearConstraintParams emptyLinParams;
  BoxConstraintParams emptyBoxParams;
  for (std::size_t c = 0; c < m_constraints.size(); ++c) {
    LinearConstraint* lin = m_constraints[c].get();
    LinearConstraintParams* lparam;
    if (lin->hasLinearInequalities()) {
      m_constraintsParams.lin.emplace_back();
      lparam = &m_constraintsParams.lin.back();
      lparam->cid = c;
    } else
      lparam = &emptyLinParams;
    BoxConstraintParams* bparam;
    if (lin->hasUbounds() || lin->hasXbounds()) {
      m_constraintsParams.box.emplace_back();
      bparam = &m_constraintsParams.box.back();
      bparam->cid = c;
    } else
      bparam = &emptyBoxParams;
    lin->computeParams(*path, times,
        lparam->a, lparam->b, lparam->c, lparam->F, lparam->g,
        bparam->u, bparam->x);
  }
}

} // namespace toppra
