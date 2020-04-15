#include <toppra/solver/qpOASES-wrapper.hpp>

#include <qpOASES.hpp>

namespace toppra {
namespace solver {
struct qpOASESWrapper::Impl {
  qpOASES::SQProblem qp;

  Impl(Eigen::Index nV, Eigen::Index nC)
    : qp (nV, nC)
  {
    qpOASES::Options options;
    options.printLevel = qpOASES::PL_NONE;

    qp.setOptions( options );
  }
};

qpOASESWrapper::qpOASESWrapper (const LinearConstraintPtrs& constraints, const GeometricPath& path,
        const Vector& times)
  : Solver (constraints, path, times)
{
  // Currently only support Canonical Linear Constraint
  Eigen::Index nC = 2; // First constraint is x + 2 D u <= xnext_max, second is xnext_min <= x + 2D u
  for (const Solver::LinearConstraintParams& linParam : constraintsParams_.lin)
    nC += linParam.F[0].rows();

  Eigen::Index nV (nbVars());
  assert(nV == 2);
  A_  = RMatrix::Zero(nC, nV);
  lA_ = -Vector::Ones(nC);
  hA_ = -Vector::Ones(nC);

  impl_ = new Impl(nV, nC);
}

qpOASESWrapper::~qpOASESWrapper ()
{
  delete impl_;
}

bool qpOASESWrapper::solveStagewiseOptim(std::size_t i,
        const Matrix& H, const Vector& g,
        const Bound& x, const Bound& xNext,
        Vector& solution)
{
  Eigen::Index N (nbStages());
  assert (i <= N);

  value_type INF = 1e16;
  Bound l (Bound::Constant(-INF)),
        h (Bound::Constant( INF));

  l[1] = std::max(l[1], x[0]);
  h[1] = std::min(h[1], x[1]);

  if (i < N) {
    value_type delta = deltas()[i];
    // TODO self._A[0] access 0-th row ?
    A_.row(0) << -2 * delta, -1;
    hA_[0] = - xNext[0];
    lA_[0] = - INF;

    // TODO self._A[1] access 1-th row ?
    A_.row(1).setZero();
    hA_[1] = xNext[1];
    lA_[1] = -INF;
  }
  Eigen::Index cur_index = 2;
  for (const Solver::LinearConstraintParams& lin : constraintsParams_.lin)
  {
    std::size_t j (lin.F.size() == 1 ? 0 : i);
    const Matrix& _F (lin.F[j]);
    const Vector& _g (lin.g[j]);
    Eigen::Index nC (_F.rows());

    A_.block(cur_index, 0, nC, 1) = _F * lin.a[i];
    A_.block(cur_index, 1, nC, 1) = _F * lin.b[i];
    hA_.segment(cur_index, nC) = _g - _F * lin.c[i];
    lA_.segment(cur_index, nC).setConstant(-INF);
    cur_index += nC;
  }
  for (const Solver::BoxConstraintParams& box : constraintsParams_.box)
  {
    if (!box.u.empty()) {
      l[0] = std::max(l[0], box.u[i][0]);
      h[0] = std::min(h[0], box.u[i][1]);
    }
    if (!box.x.empty()) {
      l[1] = std::max(l[1], box.x[i][0]);
      h[1] = std::min(h[1], box.x[i][1]);
    }
  }

  qpOASES::returnValue res;
  // TODO I assumed 1000 is the argument nWSR of the SQProblem.init function.
  //res = self.solver.init(
  //    H, g, self._A, l, h, self._lA, self._hA, np.array([1000])
  //)
  int nWSR = 1000;
  if (H.size() == 0) {
    impl_->qp.setHessianType(qpOASES::HST_ZERO);
    res = impl_->qp.init (NULL, g.data(),
        A_.data(),
        l.data(), h.data(),
        lA_.data(), hA_.data(),
        nWSR);
  } else {
    H_ = H; // Convert to row-major
    res = impl_->qp.init (H_.data(), g.data(),
        A_.data(),
        l.data(), h.data(),
        lA_.data(), hA_.data(),
        nWSR);
  }

  if (res == qpOASES::SUCCESSFUL_RETURN) {
    solution.resize(nbVars());
    impl_->qp.getPrimalSolution(solution.data());
    return true;
  }
  return false;
}

} // namespace solver
} // namespace toppra
