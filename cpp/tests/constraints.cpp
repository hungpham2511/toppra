#include <toppra/constraint/linear_joint_velocity.hpp>

#include "gtest/gtest.h"

TEST(Constraints, LinearJointVelocity) {
  using namespace toppra;
  int nDof = 5;
  Vector lb (-Vector::Ones(nDof)),
         ub ( Vector::Ones(nDof));
  constraint::LinearJointVelocity ljv (lb, ub);

  EXPECT_TRUE(ljv.constantF());
  EXPECT_EQ(ljv.discretizationType(), Collocation);
  EXPECT_EQ(ljv.nbConstraints(), 2);
  EXPECT_EQ(ljv.nbVariables(), 2);

  GeometricPath path;
  int N = 10;
  Vector gridpoints (N+1);
  {
    Vectors a(N+1, Vector(ljv.nbVariables())),
            b(N+1, Vector(ljv.nbVariables())),
            c(N+1, Vector(ljv.nbVariables())),
            g(N+1, Vector(ljv.nbConstraints()));
    Matrices F;
    EXPECT_THROW(ljv.computeParams(path, gridpoints, a, b, c, F, g), std::invalid_argument);
    F.resize(1, Matrix(ljv.nbConstraints(), ljv.nbVariables()));
    EXPECT_THROW(ljv.computeParams(path, gridpoints, a, b, c, F, g), std::logic_error);
  }

  {
    Bounds ub, xb;
    EXPECT_THROW(ljv.computeBounds(path, gridpoints, ub, xb), std::invalid_argument);
    xb.resize(N+1);
    EXPECT_NO_THROW(ljv.computeBounds(path, gridpoints, ub, xb));
  }
}

