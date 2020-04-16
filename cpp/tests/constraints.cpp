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
  EXPECT_EQ(ljv.nbConstraints(), 0);
  EXPECT_EQ(ljv.nbVariables(), 0);

  GeometricPath path;
  int N = 10;
  Vector gridpoints (N+1);
  {
    Vectors a, b, c, g;
    Matrices F;
    Bounds ub, xb;
    EXPECT_THROW(ljv.computeParams(path, gridpoints, a, b, c, F, g, ub, xb), std::invalid_argument);
    ljv.allocateParams(gridpoints.size(), a, b, c, F, g, ub, xb);
    EXPECT_EQ(a .size(), 0);
    EXPECT_EQ(b .size(), 0);
    EXPECT_EQ(c .size(), 0);
    EXPECT_EQ(F .size(), 0);
    EXPECT_EQ(g .size(), 0);
    EXPECT_EQ(ub.size(), 0);
    EXPECT_EQ(xb.size(), N+1);
    EXPECT_NO_THROW(ljv.computeParams(path, gridpoints, a, b, c, F, g, ub, xb));
  }
}

