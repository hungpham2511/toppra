#include <toppra/constraint/linear_joint_velocity.hpp>

#ifdef BUILD_WITH_PINOCCHIO
#include <toppra/constraint/joint_torque/pinocchio.hpp>
#include <pinocchio/parsers/sample-models.hpp>
#endif

#include <toppra/geometric_path.hpp>

#include "gtest/gtest.h"

class Constraint : public testing::Test {
public:
  Constraint() : path(constructPath()) {

    // path has the equation: 0 * x ^ 3 + 1 * x ^ 2 + 2 x ^ 1 + 3
  }

  toppra::PiecewisePolyPath constructPath() {
    toppra::Matrix coeff{4, 6};
    coeff.colwise() = toppra::Vector::LinSpaced(4, 0, 3);
    toppra::Matrices coefficents = {coeff, coeff};
    toppra::PiecewisePolyPath p =
        toppra::PiecewisePolyPath(coefficents, std::vector<double>{0, 1, 2});
    return p;
  }

  toppra::PiecewisePolyPath path;
};

TEST_F(Constraint, LinearJointVelocity) {
  using namespace toppra;
  int nDof = path.dof();
  Vector lb (-Vector::Ones(nDof)),
         ub ( Vector::Ones(nDof));
  constraint::LinearJointVelocity ljv (lb, ub);

  EXPECT_TRUE(ljv.constantF());
  EXPECT_EQ(ljv.discretizationType(), Collocation);
  EXPECT_EQ(ljv.nbConstraints(), 0);
  EXPECT_EQ(ljv.nbVariables(), 0);

  int N = 10;
  Vector gridpoints;
  {
    Bound I (path.pathInterval());
    gridpoints = toppra::Vector::LinSpaced (N, I[0], I[1]);
  }
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
    EXPECT_EQ(xb.size(), N);
    EXPECT_NO_THROW(ljv.computeParams(path, gridpoints, a, b, c, F, g, ub, xb));
  }
}

#ifdef BUILD_WITH_PINOCCHIO
TEST_F(Constraint, jointTorquePinocchio) {
  using namespace toppra;
  typedef constraint::jointTorque::Pinocchio<> JointTorque;

  JointTorque::Model model;
  pinocchio::buildModels::manipulator(model);
  Vector frictions (Vector::Constant(model.nv, 0.001));
  JointTorque constraint (model, frictions);

  EXPECT_EQ(constraint.nbVariables(), model.nv);
  EXPECT_EQ(constraint.nbConstraints(), 2*model.nv);

  int N = 10;
  Vector gridpoints;
  {
    Bound I (path.pathInterval());
    gridpoints = toppra::Vector::LinSpaced (N, I[0], I[1]);
  }

  {
    Vectors a, b, c, g;
    Matrices F;
    Bounds ub, xb;
    EXPECT_THROW(constraint.computeParams(path, gridpoints, a, b, c, F, g, ub, xb), std::invalid_argument);
    constraint.allocateParams(gridpoints.size(), a, b, c, F, g, ub, xb);
    EXPECT_NO_THROW(constraint.computeParams(path, gridpoints, a, b, c, F, g, ub, xb));
  }
}
#endif
