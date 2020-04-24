#include <gmock/gmock-matchers.h>
#include <memory>
#include <toppra/algorithm/toppra.hpp>
#include <toppra/constraint.hpp>
#include <toppra/constraint/linear_joint_acceleration.hpp>
#include <toppra/constraint/linear_joint_velocity.hpp>
#include <toppra/geometric_path.hpp>
#include <toppra/toppra.hpp>
#include "toppra/algorithm.hpp"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

class ProblemInstance : public testing::Test {
 public:
  ProblemInstance() : path(constructPath()) {
    // path has the equation: 0 * x ^ 3 + 1 * x ^ 2 + 2 x ^ 1 + 3
  }

  toppra::PiecewisePolyPath constructPath() {
    toppra::Matrix coeff{4, 2};
    coeff(0, 0) = 0;
    coeff(1, 0) = 1;
    coeff(2, 0) = 2;
    coeff(3, 0) = 3;
    coeff(0, 1) = 0;
    coeff(1, 1) = 1;
    coeff(2, 1) = 2;
    coeff(3, 1) = 3;
    toppra::Matrices coefficents = {coeff};
    toppra::PiecewisePolyPath p =
        toppra::PiecewisePolyPath(coefficents, std::vector<double>{0, 1});
    return p;
  }

  toppra::Vector getTimes(int N) {
    toppra::Bound I(path.pathInterval());
    return toppra::Vector::LinSpaced(N, I[0], I[1]);
  }

  toppra::PiecewisePolyPath path;
  int nDof = 2;
};

TEST_F(ProblemInstance, DISABLED_ConstructNewInstance) {
  toppra::LinearConstraintPtrs v{
      std::make_shared<toppra::constraint::LinearJointVelocity>(
          -toppra::Vector::Ones(nDof), toppra::Vector::Ones(nDof)),
      std::make_shared<toppra::constraint::LinearJointAcceleration>(
          -0.2 * toppra::Vector::Ones(nDof), 0.2 * toppra::Vector::Ones(nDof))};
  toppra::algorithm::TOPPRA instance{v, path};
  toppra::ReturnCode ret_code = instance.computePathParametrization();

  ASSERT_THAT(ret_code, toppra::ReturnCode::OK)
      << "actual return code: " << (int)ret_code;
}
