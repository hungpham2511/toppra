#include <toppra/solver.hpp>

#include <toppra/constraint.hpp>
#include <toppra/geometric_path.hpp>

namespace toppra {
void Solver::LinearConstraintParams::init(std::size_t N, LinearConstraint* lin)
{
  Eigen::Index nv (lin->nbVariables());
  Eigen::Index nc (lin->nbConstraints());
  a.resize(N+1, Vector(nv));
  b.resize(N+1, Vector(nv));
  c.resize(N+1, Vector(nv));
  if (lin->constantF()) {
    F.resize(1, Matrix(nc, nv));
    g.resize(1, Vector(nc));
  } else {
    F.resize(N+1, Matrix(nc, nv));
    g.resize(N+1, Vector(nc));
  }
}

void Solver::BoxConstraintParams::init(std::size_t N, BoxConstraint* box)
{
  if (box->hasUbounds()) u.resize(N+1);
  if (box->hasXbounds()) x.resize(N+1);
}

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
  for (std::size_t c = 0; c < constraints_.size(); ++c) {
    LinearConstraint* lin = constraints_[c].get();
    BoxConstraint* box = dynamic_cast<BoxConstraint*> (lin);
    if (box) {
      constraintsParams_.box.emplace_back();
      BoxConstraintParams& param = constraintsParams_.box.back();
      param.cid = c;
      param.init(N_, box);
      box->computeBounds(path, times, param.u, param.x);
    } else {
      constraintsParams_.lin.emplace_back();
      LinearConstraintParams& param = constraintsParams_.lin.back();
      param.cid = c;
      param.init(N_, lin);
      lin->computeParams(path, times, param.a, param.b, param.c, param.F, param.g);
    }
  }
}

} // namespace toppra
