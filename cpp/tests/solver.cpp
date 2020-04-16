#include <toppra/solver/qpOASES-wrapper.hpp>

#include <toppra/constraint/linear_joint_velocity.hpp>
#include <toppra/constraint/linear_joint_acceleration.hpp>

#include <toppra/geometric_path.hpp>

#include "gtest/gtest.h"

#ifdef BUILD_WITH_qpOASES
TEST(Solver, qpOASESWrapper) {
  using namespace toppra;
  int nDof = 5;
  Vector lb (-Vector::Ones(nDof)),
         ub ( Vector::Ones(nDof));
  LinearConstraintPtr ljv (new constraint::LinearJointVelocity (lb, ub));
  LinearConstraintPtr lja (new constraint::LinearJointAcceleration (lb, ub));

  GeometricPath path;
  int N = 10;
  Vector times (N);
  for (int i = 0; i < N; ++i) times[i] = .01 * i;
  solver::qpOASESWrapper solver ({ ljv, lja }, path, times);

  EXPECT_EQ(solver.nbStages(), N-1);
  EXPECT_EQ(solver.nbVars(), 2);
  ASSERT_EQ(solver.deltas().size(), N-1);
  for (int i = 0; i < N-1; ++i)
    EXPECT_NEAR(solver.deltas()[i], 0.01, 1e-10);
}
#endif
