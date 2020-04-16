#include <toppra/solver.hpp>

#include <toppra/constraint.hpp>
#include <toppra/geometric_path.hpp>

namespace toppra {

Solver::Solver (const LinearConstraintPtrs& constraints, const GeometricPath& path,
    const Vector& times)
  : constraints_ (constraints)
  , path_ (path)
  , times_ (times)
  , N_ (times.size()-1)
  , nV_ (2)
  , deltas_ (times.tail(N_) - times.head(N_))
{
  if ((deltas_.array() <= 0).any())
    throw std::invalid_argument("Invalid times.");
  /// \todo assert that the time range of the path equals [ times[0], times[N_] ].

  // Compute the constraints parameters
  LinearConstraintParams emptyLinParams;
  BoxConstraintParams emptyBoxParams;
  for (std::size_t c = 0; c < constraints_.size(); ++c) {
    LinearConstraint* lin = constraints_[c].get();
    LinearConstraintParams* lparam;
    if (lin->hasLinearInequalities()) {
      constraintsParams_.lin.emplace_back();
      lparam = &constraintsParams_.lin.back();
      lparam->cid = c;
    } else
      lparam = &emptyLinParams;
    BoxConstraintParams* bparam;
    if (lin->hasUbounds() || lin->hasXbounds()) {
      constraintsParams_.box.emplace_back();
      bparam = &constraintsParams_.box.back();
      bparam->cid = c;
    } else
      bparam = &emptyBoxParams;
    lin->allocateParams(times.size(),
        lparam->a, lparam->c, lparam->b, lparam->F, lparam->g,
        bparam->u, bparam->x);
    lin->computeParams(path, times,
        lparam->a, lparam->c, lparam->b, lparam->F, lparam->g,
        bparam->u, bparam->x);
  }
}

} // namespace toppra
