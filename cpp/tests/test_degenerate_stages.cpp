#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
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

// Times spans with 32 collocation stages each from `vel_start` (squared) to
// `vel_end`, and checks that the result is within the controllable sets, the
// velocity limits, and the acceleration limits of every joint but the last.
void checkSpans(const Matrices& coefficients, const std::vector<value_type>& knots,
                const Vector& velocity, const Vector& acceleration,
                value_type vel_start, value_type vel_end) {
  auto path = std::make_shared<PiecewisePolyPath>(coefficients, knots);
  LinearConstraintPtrs constraints{
    std::make_shared<constraint::LinearJointVelocity>(-velocity, velocity),
    std::make_shared<constraint::LinearJointAcceleration>(-acceleration, acceleration)};
  for (auto& c : constraints)
    c->discretizationType(DiscretizationType::Collocation);

  const int spans = static_cast<int>(coefficients.size()), N = 32 * spans;
  Vector gridpoints(N + 1);
  for (int span = 0; span < spans; ++span)
    for (int k = 0; k < 32; ++k)
      gridpoints[32 * span + k] = knots[span] + (knots[span + 1] - knots[span]) * k / 32.0;
  gridpoints[N] = knots[spans];

  algorithm::TOPPRA problem(constraints, path);
  problem.solver(std::make_shared<solver::Seidel>());
  problem.setGridpoints(gridpoints);
  problem.setInitialXBounds(Bound{0, 1e16});
  ASSERT_EQ(problem.computePathParametrization(vel_start, vel_end), ReturnCode::OK)
    << problem.getErrorMessage();

  // vel_start is recorded, so it may sit a rounding off the recomputed sets.
  const auto& data = problem.getParameterizationData();
  const value_type ulps = 4 * std::numeric_limits<value_type>::epsilon();
  for (int i = 0; i <= N; ++i) {
    EXPECT_GE(data.parametrization[i], data.controllable_sets(i, 0) * (1 - ulps));
    EXPECT_LE(data.parametrization[i], data.controllable_sets(i, 1) * (1 + ulps));
  }
  EXPECT_NEAR(data.parametrization[N], vel_end * vel_end, 1e-12);
  for (int i = 0; i <= N; ++i) {
    const Vector qs = path->eval_single(gridpoints[i], 1);
    for (int j = 0; j < qs.size(); ++j)
      EXPECT_LE(qs[j] * qs[j] * data.parametrization[i],
                velocity[j] * velocity[j] * (1 + 1e-9)) << "stage " << i << ", joint " << j;
  }
  for (int i = 0; i < N; ++i) {
    const value_type u = (data.parametrization[i + 1] - data.parametrization[i])
      / (2 * (gridpoints[i + 1] - gridpoints[i]));
    const Vector qs = path->eval_single(gridpoints[i], 1);
    const Vector qss = path->eval_single(gridpoints[i], 2);
    for (int j = 0; j + 1 < qs.size(); ++j)
      EXPECT_LE(std::abs(qs[j] * u + qss[j] * data.parametrization[i]),
                acceleration[j] * (1 + 1e-9)) << "stage " << i << ", joint " << j;
  }
}

void checkSpan(const Matrix& coefficients, value_type s0, value_type s1,
               const Vector& velocity, const Vector& acceleration,
               value_type vel_start, value_type vel_end) {
  checkSpans(Matrices{coefficients}, {s0, s1}, velocity, acceleration,
             vel_start, vel_end);
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

// At the upper bound of the controllable set where the forward pass starts,
// the stage admits a single control, but one joint, at rest along the path,
// has a coefficient of the path acceleration that is rounding (-1.8e-16)
// next to its curvature: solved again with the state fixed there, the stage
// has no control (the forward pass failed at stage 0).
TEST(DegenerateStages, ForwardPassFromTheUpperBoundOfAControllableSet) {
  Matrix coefficients(6, 5);
  coefficients <<
    -2370457952906959.5, 243320965084031.53, 1883815053995587.8, -1883815054702169.8, 0,
    619968412117.37317, -620245633071.05017, 620522885794.68921, -620522885771.57886, 0,
    142129377.26969227, 13922.379463409125, -142157222.3064588, 142157222.30625778, 0,
    -24557.222204312384, 24557.222204312384, -24557.22220431238, 24557.22220431238, 0,
    -2.8284271247461898, -1.8093822187881339e-16, 2.8284271247461898, -2.8284271247461907, 1,
    -0.10902681754516537, 0.035693247325073944, 0.037640322895017515, -0.037640322895017515, 0.23625306276186719;
  Vector velocity(5), acceleration(5);
  velocity << 1, 1, 1, 1, 0.059999999999999998;
  acceleration << 40.200681829243933, 58.650764514538537, 77.100847199833169, 77.100847199833169, 1e12;
  checkSpan(coefficients, 0.2362530627618672, 0.23626618074820144, velocity,
            acceleration, 0.0011941652851974265, 0.03336501157674108);
}

// Holding the feed rate into a curve whose curvature limit is the upper bound
// of the controllable sets: the step into stage 32 ended one rounding below
// that bound instead of on it, where the stage is as degenerate as on it, and
// the forward pass failed there.
TEST(DegenerateStages, ForwardPassStepEndingOneRoundingBelowTheBound) {
  Matrices coefficients(3, Matrix(6, 6));
  coefficients[0] <<
    -0.00731368052024886, -0.0024734896923367692, -0.022348961414256206, 0.022348961414256206, -0.022325972114443888, 0,
    3.7776198808741251e-05, 9.5430908321258987e-06, 0.0001056421768004828, -0.0001056421768004828, 0.0001055128311554497, 0,
    -5.493716764733474e-08, -6.0767980873115041e-09, -1.2999975105867079e-07, 1.2999975105867079e-07, -1.2978605981313577e-07, 0,
    0, 0, 0, 0, 0, 0,
    -2.8284271247461898, 0, -2.8284271247461898, 2.8284271247461898, -2.8284271247461898, 1,
    -0.094488989463560435, 0.02878487687505963, -0.036919235713441134, 0.036919235713441134, -0.036919235713441134, 0.26940906365862388;
  coefficients[1] <<
    -262437176940018.84, 28814601672781.379, -204807986276830.75, 204807986276830.75, -204807986932763.19, 0,
    118469904176.65001, -118536104238.72847, -118602303522.55927, 118602303522.55927, -118602303482.32393, 0,
    47129687.181422733, 5769.9697412705636, 47141227.108169794, -47141227.108169794, 47141227.1075119, 0,
    -14141.294984656979, 14141.294984657154, 14141.294984657328, -14141.294984657328, 14141.294984657328, 0,
    -2.8284271247462249, 3.4921076822610987e-14, -2.8284271247461548, 2.8284271247461548, -2.8284271247461552, 1,
    -0.099958454089538074, 0.028784876875059654, -0.042388700339418772, 0.042388700339418772, -0.042388700339418772, 0.27134281142186822;
  coefficients[2] <<
    -304336546491276.19, 84716701714344.047, -134903112719494.56, 134903112719494.56, -134903112390244.28, 0,
    86101500659.994598, -114967639346.73045, -143833779895.41638, 143833779895.41638, -143833779915.61008, 0,
    57171653.361353323, -11447384.825333232, 34276883.741155058, -34276883.741155058, 34276883.741485186, 0,
    -10281.44369632266, 13717.498095760517, 17153.552495198368, -17153.552495198368, 17153.552495198372, 0,
    -3.4309144400011986, 0.68725173161424813, -2.0564109767727028, 2.0564109767727028, -2.0564109767727028, 1,
    -0.10003566029363888, 0.028793353820583529, -0.042448952652471846, 0.042448952652471846, -0.042448952652471846, 0.2713673565734912;
  Vector velocity(6), acceleration(6);
  velocity << 1, 1, 1, 1, 1, 0.059999999999999998;
  acceleration << 40.200681829243933, 58.650764514538537, 77.100847199833169,
    77.100847199833169, 77.100847199833169, 1e12;
  checkSpans(coefficients,
             {0.2694090636586239, 0.2713428114218682, 0.2713673565734912, 0.2713919017251142},
             velocity, acceleration, 0, 0.04438962564361016);
}

namespace {
// Two lines joined by a circular arc that deviates at most `deviation` from
// the corner, parametrized by arc length (roboplan's LinearBlendPath).
class CornerBlendPath : public GeometricPath {
 public:
  CornerBlendPath(const Vector& a, const Vector& b, const Vector& c, value_type deviation)
      : GeometricPath(static_cast<int>(a.size())) {
    const Vector in = b - 0.5 * (a + b), out = 0.5 * (b + c) - b;
    const Vector u_in = in.normalized(), u_out = out.normalized();
    const value_type half = 0.5 * std::acos(std::min(std::max(u_in.dot(u_out), -1.0), 1.0));
    const value_type dist = std::min({in.norm(), out.norm(),
        deviation * std::sin(half) / (1.0 - std::cos(half))});
    m_radius = dist / std::tan(half);
    m_center = b + (u_out - u_in).normalized() * (m_radius / std::cos(half));
    m_x = (b - dist * u_in - m_center).normalized();
    m_y = u_in;
    m_line0 = {a, b - dist * u_in};
    m_line1 = {b + dist * u_out, c};
    m_knots << 0, (m_line0[1] - a).norm(), 0, 0;
    m_knots[2] = m_knots[1] + 2 * half * m_radius;
    m_knots[3] = m_knots[2] + (c - m_line1[0]).norm();
  }

  Vector eval_single(value_type s, int order = 0) const override {
    s = std::min(std::max(s, 0.0), m_knots[3]);
    if (s < m_knots[1] || s >= m_knots[2]) {
      const auto& l = s < m_knots[1] ? m_line0 : m_line1;
      const value_type len = (l[1] - l[0]).norm(), u = s - m_knots[s < m_knots[1] ? 0 : 2];
      const Vector d = (l[1] - l[0]) / len;
      if (order == 0) return l[0] + u * d;
      if (order == 1) return d;
      return Vector::Zero(configSize());
    }
    const value_type t = (s - m_knots[1]) / m_radius;
    if (order == 0) return m_center + m_radius * (m_x * std::cos(t) + m_y * std::sin(t));
    if (order == 1) return -m_x * std::sin(t) + m_y * std::cos(t);
    if (order == 2) return (-1.0 / m_radius) * (m_x * std::cos(t) + m_y * std::sin(t));
    return Vector::Zero(configSize());
  }

  Bound pathInterval() const override { return Bound(0, m_knots[3]); }
  Vector knots() const { return m_knots; }

 private:
  value_type m_radius;
  Vector m_center, m_x, m_y;
  std::array<Vector, 2> m_line0, m_line1;
  Eigen::Vector4d m_knots;
};
}  // namespace

// From https://gist.github.com/sea-bass/48cacaee82c9f6c0c61faf93b6c34c71
// (case_00_franka): the forward pass reaches the upper bound of stage 120's
// controllable set, a vertex where a row's path-acceleration coefficient is
// 5.6e-5, and solved again there the stage had no control.
TEST(DegenerateStages, ForwardPassAtTheUpperBoundOnACornerBlend) {
  Vector a(7), b(7), c(7);
  a << -1.4964286529696755, -0.92757880736655007, 0.35597239013785331, -1.6613642304192493,
       2.0448851059425603, 3.3599181331464085, -1.2694354413179221;
  b << -1.349862036160935, -0.80024976155651761, 0.36305347219510592, -1.4981976182525667,
       1.9981188101837097, 3.4894267267580341, -1.3750672864107198;
  c << -1.238434278003687, -0.67717180344092609, 0.37218371195179278, -1.3074581491277963,
       2.0180510666453548, 3.5976213885093054, -1.5356673978691857;
  auto path = std::make_shared<CornerBlendPath>(a, b, c, 0.00509585);
  const Vector ones = Vector::Ones(7);
  auto velocity = std::make_shared<constraint::LinearJointVelocity>(-3.14 * ones, 3.14 * ones);
  auto acceleration = std::make_shared<constraint::LinearJointAcceleration>(-2.0 * ones, 2.0 * ones);
  acceleration->discretizationType(DiscretizationType::Interpolation);

  algorithm::TOPPRA problem({velocity, acceleration}, path);
  problem.setGridpoints(path->proposeGridpoints(1e-5, 100, 0.02, 100, path->knots()));
  EXPECT_EQ(problem.computePathParametrization(), ReturnCode::OK) << problem.getErrorMessage();
}
