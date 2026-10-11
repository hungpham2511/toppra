#include <memory>
#include <toppra/algorithm/toppra.hpp>
#include <toppra/constraint/linear_joint_acceleration.hpp>
#include <toppra/constraint/linear_joint_velocity.hpp>
#include <toppra/geometric_path/piecewise_poly_path.hpp>
#include <toppra/solver/seidel.hpp>
#include <toppra/toppra.hpp>

#include "gtest/gtest.h"

// Spans recorded from a path planner, on which TOPPRA with the Seidel solver
// failed although the paths are feasible. Each has one column per joint,
// scaled so that its velocity limit is 1, and a last column that is the path
// parameter itself, whose velocity limit is the feed rate. Two of the joints
// have nearly identical constraints.

namespace {
using namespace toppra;

// Times one span with 32 collocation stages from `vel_start` (squared) to
// `vel_end`, and checks that the result is within the controllable sets and
// the acceleration limits of every joint but the last.
void checkSpan(const Matrix& coefficients, value_type s0, value_type s1,
               const Vector& velocity, const Vector& acceleration,
               value_type vel_start, value_type vel_end) {
  auto path = std::make_shared<PiecewisePolyPath>(Matrices{coefficients},
                                                  std::vector<value_type>{s0, s1});
  LinearConstraintPtrs constraints{
    std::make_shared<constraint::LinearJointVelocity>(-velocity, velocity),
    std::make_shared<constraint::LinearJointAcceleration>(-acceleration, acceleration)};
  for (auto& c : constraints)
    c->discretizationType(DiscretizationType::Collocation);

  Vector gridpoints(33);
  for (int k = 0; k < 32; ++k) gridpoints[k] = s0 + (s1 - s0) * k / 32.0;
  gridpoints[32] = s1;

  algorithm::TOPPRA problem(constraints, path);
  problem.solver(std::make_shared<solver::Seidel>());
  problem.setGridpoints(gridpoints);
  problem.setInitialXBounds(Bound{0, 1e16});
  ASSERT_EQ(problem.computePathParametrization(vel_start, vel_end), ReturnCode::OK)
    << problem.getErrorMessage();

  const auto& data = problem.getParameterizationData();
  for (int i = 0; i < 33; ++i) {
    EXPECT_GE(data.parametrization[i], data.controllable_sets(i, 0));
    EXPECT_LE(data.parametrization[i], data.controllable_sets(i, 1));
  }
  EXPECT_NEAR(data.parametrization[32], vel_end * vel_end, 1e-12);
  for (int i = 0; i < 32; ++i) {
    const value_type u = (data.parametrization[i + 1] - data.parametrization[i])
      / (2 * (gridpoints[i + 1] - gridpoints[i]));
    const Vector qs = path->eval_single(gridpoints[i], 1);
    const Vector qss = path->eval_single(gridpoints[i], 2);
    for (int j = 0; j + 1 < qs.size(); ++j)
      EXPECT_LE(std::abs(qs[j] * u + qss[j] * data.parametrization[i]),
                acceleration[j] * (1 + 1e-9)) << "stage " << i << ", joint " << j;
  }
}
}  // namespace

// Decelerating at the limit along the upper bounds of the controllable sets.
// The forward pass failed at stage 22: on the line of one joint's constraint,
// the other's became a bound placed by the rounding of its offset alone.
TEST(DegenerateStages, ForwardPassAlongNearlyParallelConstraints) {
  // Highest degree first, as PiecewisePolyPath expects.
  Matrix coefficients(6, 3);
  coefficients <<
    -0.035100294744607873, -0.035583317494249894, 0,
    9.0185246504736075e-05, 9.1339342949361076e-05, 0,
    -5.9182501316889105e-08, -5.9811175528834993e-08, 0,
    0, 0, 0,
    -3.778739903238066, -3.778739903238066, 1,
    -0.032902774415385541, -0.032902774415385541, 0.43946669291217472;
  Vector velocity(3), acceleration(3);
  velocity << 1, 1, 0.14999999999999999;
  acceleration << 77.100847199833169, 77.100847199833169, 1e12;
  checkSpan(coefficients, 0.4394666929121747, 0.4405346793459111, velocity,
            acceleration, 0, 0.02);
}

// The same in the backward pass: the controllable set at stage 18 could not
// be computed (ERR_FAIL_CONTROLLABLE).
TEST(DegenerateStages, ControllableSetsAlongNearlyParallelConstraints) {
  Matrix coefficients(6, 4);
  coefficients <<
    2665713060502331, 15623518820471.861, -15623519407697.836, 0,
    -94213934419.284897, 791593552940.99487, -791593552921.23389, 0,
    -187283739.71060124, -22391348.273390654, 22391348.273213021, 0,
    3990.3853320244684, -33403.165286311829, 33403.165286311836, 0,
    3.9717599026261001, 0.47447157543051072, -0.47447157543051072, 1,
    0.016779113593640864, -0.00078797569681881392, 0.00078797569681881392, 0.11183855169944042;
  Vector velocity(4), acceleration(4);
  velocity << 1, 1, 1, 0.060000000000000005;
  acceleration << 40.200681829243933, 77.100847199833169, 77.100847199833169, 1e12;
  checkSpan(coefficients, 0.11183855169944042, 0.1118519922150172, velocity,
            acceleration, 0.0010668838839529022, 0.033);
}
