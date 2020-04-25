#include <toppra/solver/qpOASES-wrapper.hpp>
#include <toppra/solver/glpk-wrapper.hpp>

#include <toppra/constraint/linear_joint_velocity.hpp>
#include <toppra/constraint/linear_joint_acceleration.hpp>

#include <toppra/geometric_path.hpp>

#include "gtest/gtest.h"

class Solver : public testing::Test {
public:
  Solver() : path(constructPath()) {

    // path has the equation: 0 * x ^ 3 + 1 * x ^ 2 + 2 x ^ 1 + 3
  }

  toppra::PiecewisePolyPath constructPath() {
    toppra::Matrix coeff{4, 2};
    coeff.colwise() = toppra::Vector::LinSpaced(4, 0, 3);
    toppra::Matrices coefficents = {coeff, coeff};
    toppra::PiecewisePolyPath p =
        toppra::PiecewisePolyPath(coefficents, std::vector<double>{0, 1, 2});
    return p;
  }

  toppra::Vector getTimes (int N)
  {
    toppra::Bound I (path.pathInterval());
    return toppra::Vector::LinSpaced (N, I[0], I[1]);
  }

  toppra::PiecewisePolyPath path;
};

#ifdef BUILD_WITH_qpOASES
TEST_F(Solver, qpOASESWrapper) {
  using namespace toppra;
  int nDof = path.dof();
  Vector lb (-Vector::Ones(nDof)),
         ub ( Vector::Ones(nDof));
  LinearConstraintPtr ljv (new constraint::LinearJointVelocity (lb, ub));
  LinearConstraintPtr lja (new constraint::LinearJointAcceleration (lb, ub));

  int N = 10;
  Vector times (getTimes(N));
  solver::qpOASESWrapper solver ({ ljv, lja }, path, times);

  EXPECT_EQ(solver.nbStages(), N-1);
  EXPECT_EQ(solver.nbVars(), 2);
  ASSERT_EQ(solver.deltas().size(), N-1);
  for (int i = 0; i < N-1; ++i)
    EXPECT_NEAR(solver.deltas()[i], times[i+1] - times[i], 1e-10);

  solver.setupSolver();
  Vector g (Vector::Ones(2)), solution;
  Matrix H;
  const value_type infty (std::numeric_limits<value_type>::infinity());
  Bound x, xNext;
  x << -infty, infty;
  xNext << -infty, infty;
  for (int i = 0; i < N; ++i) {
    EXPECT_TRUE(solver.solveStagewiseOptim(i, H, g, x, xNext, solution));
    EXPECT_EQ(solution.size(), solver.nbVars());
  }
  solver.closeSolver();
}
#endif

#ifdef BUILD_WITH_GLPK
TEST_F(Solver, GLPKWrapper) {
  using namespace toppra;
  int nDof = path.dof();
  Vector lb (-Vector::Ones(nDof)),
         ub ( Vector::Ones(nDof));
  LinearConstraintPtr ljv (new constraint::LinearJointVelocity (lb, ub));
  LinearConstraintPtr lja (new constraint::LinearJointAcceleration (lb, ub));

  int N = 10;
  Vector times (getTimes(N));
  solver::GLPKWrapper solver ({ ljv, lja }, path, times);

  EXPECT_EQ(solver.nbStages(), N-1);
  EXPECT_EQ(solver.nbVars(), 2);
  ASSERT_EQ(solver.deltas().size(), N-1);
  for (int i = 0; i < N-1; ++i)
    EXPECT_NEAR(solver.deltas()[i], times[i+1] - times[i], 1e-10);

  solver.setupSolver();
  Vector g (Vector::Ones(2)), solution;
  Matrix H;
  const value_type infty (std::numeric_limits<value_type>::infinity());
  Bound x, xNext;
  x << -infty, infty;
  xNext << -infty, infty;
  for (int i = 0; i < N; ++i) {
    EXPECT_TRUE(solver.solveStagewiseOptim(i, H, g, x, xNext, solution));
    EXPECT_EQ(solution.size(), solver.nbVars());
  }
  solver.closeSolver();
}
#endif
