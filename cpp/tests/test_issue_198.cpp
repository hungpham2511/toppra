#include "Eigen/Dense"
#include <gtest/gtest.h>
#include <iostream>


#include <toppra/algorithm/toppra.hpp>
#include <toppra/constraint/linear_joint_acceleration.hpp>
#include <toppra/constraint/linear_joint_velocity.hpp>
#include <toppra/geometric_path/piecewise_poly_path.hpp>
#include <toppra/parametrizer/const_accel.hpp>
#include <toppra/parametrizer/spline.hpp>
#include <toppra/toppra.hpp>


#define PRINT(X) std::cout << X << std::endl


// test case obtained from issue 198
namespace test {
using namespace toppra;
class ToppraAlgorithmTest : public ::testing::Test {};

TEST_F(ToppraAlgorithmTest, TestTrajectoryTime) {
  using Eigen::VectorXd;
  // Number of samples.
  const int kNumSamples = 10;
  const int kNumJoints = 10;

  // Joint positions.
  VectorXd q1(kNumSamples);
  q1 << 0.3609483414448889, 0.529632298715096, 0.8206167110829612, 0.4261921958157011, 0.03521854768551276,
      0.9428425587520547, 0.9436612509679243, 0.7497121976441605, 0.4241365697401913, -0.03568596575328783;
  VectorXd q2(kNumSamples);
  q2 << 0.360201952956941, 0.5291881925375327, 0.8197786060659787, 0.5242293711803413, 0.03429311024704686,
      0.943238812972426, 0.9428699658756429, 0.7488763297865405, 0.5221709348328234, -0.03476051297357887;
  VectorXd q3(kNumSamples);
  q3 << 0.2576160398680044, 0.4464623529671758, 0.6181800430413302, 0.5273042719577605, 0.03410712221265201,
      0.9711494838346122, 0.7694553462390827, 0.5472782363447702, 0.5252360928004322, -0.03457452185426948;
  VectorXd q4(kNumSamples);
  q4 << 0.2932441944501117, 0.4032251888107473, 0.4069862459048218, 0.5274397139191792, 0.03397464628167379,
      0.8784834270554451, 0.617219148369562, 0.3360847290275261, 0.525364267067897, -0.03444204372337609;
  VectorXd q5(kNumSamples);
  q5 << 0.4272847882088048, 0.4062873126043486, 0.1957934375102555, 0.527585075109299, 0.03383074847742296,
      0.7114993411065769, 0.501742908489109, 0.12489225045979, 0.5255052520124111, -0.03429814353063214;
  VectorXd q6(kNumSamples);
  q6 << 0.6112115060535566, 0.4555108824340764, -0.01539832359963351, 0.5277338952480328, 0.03368182418073429,
      0.5202073666339974, 0.4281621331482424, -0.08629915572294676, 0.5256527817877672, -0.0341492167646652;
  VectorXd q7(kNumSamples);
  q7 << 0.7962614752743197, 0.5487159499360755, -0.2265890264379919, 0.5278795622940541, 0.03353449007591972,
      0.3536991431623716, 0.3997201441986077, -0.2974894943573977, 0.525800301703242, -0.03400188021985093;
  VectorXd q8(kNumSamples);
  q8 << 0.9329791033473422, 0.6817548358001362, -0.437778706855262, 0.5280156023996297, 0.03339529400753713,
      0.260613040199016, 0.4177452953668706, -0.5086788173297849, 0.5259412555966261, -0.03386268184819357;
  VectorXd q9(kNumSamples);
  q9 << 0.9708760225336093, 0.8487186653349669, -0.6489674447638665, 0.5281359713696465, 0.03327042029913405,
      0.2894909766162013, 0.4813773226209204, -0.7198672201061661, 0.5260693808376938, -0.03373780607340813;
  VectorXd q10(kNumSamples);
  q10 << 0.9388822987200556, 0.9386816585668716, -0.7501796157119357, 0.4261921958155511, 0.03521854768551277,
      0.3608387490194704, 0.527342558756185, -0.821084129150736, 0.4241365697402097, -0.03568596575328783;

  // Joint Velocities.
  VectorXd q1_dot(kNumSamples);
  q1_dot << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
  VectorXd q2_dot(kNumSamples);
  q2_dot << -0.03191916351778297, -0.01774472296483642, -0.03504316252942192, 0.0929168627602453, -0.002104201007544407,
      0.01739682283147941, -0.03215065241678145, -0.03503975153876462, 0.09291451831402819, 0.002104235887623129;
  VectorXd q3_dot(kNumSamples);
  q3_dot << -0.03003698929845297, -0.06861251724646687, -0.21995346327039, 0.000133745689670505, -0.0001299139769254685,
      -0.03979304861924536, -0.1754599413042045, -0.219953187059946, 0.0001247718267154291, 0.0001299161340127446;
  VectorXd q4_dot(kNumSamples);
  q4_dot << 0.09688740575526912, -0.02120313280832103, -0.2199524803660356, 0.000147322081645501,
      -0.0001449984757960818, -0.1440551595548057, -0.1404634572188314, -0.2199521552138202, 0.0001412136601310098,
      0.0001450008834769401;
  VectorXd q5_dot(kNumSamples);
  q5_dot << 0.1740140007363397, 0.02745541078635533, -0.2199514118857625, 0.0001543364290795491, -0.0001536233637190059,
      -0.1951590528442481, -0.09918654642616624, -0.2199510526156652, 0.0001513635877069163, 0.0001536259124344061;
  VectorXd q6_dot(kNumSamples);
  q6_dot << 0.200653666869634, 0.07472455154879661, -0.2199503071682509, 0.0001544968859816527, -0.0001554261047258265,
      -0.194791129327524, -0.05352145296446436, -0.2199499298990992, 0.0001547919765335344, 0.0001554286802232283;
  VectorXd q7_dot(kNumSamples);
  q7_dot << 0.1762631596141885, 0.1187111669649581, -0.2199492156850364, 0.0001477963131879767, -0.0001503264729818309,
      -0.1436691483824442, -0.005711316177666173, -0.2199488371205753, 0.0001513460551790628, 0.0001503289612269195;
  VectorXd q8_dot(kNumSamples);
  q8_dot << 0.09965818520438458, 0.1573754021112097, -0.2199481852719111, 0.0001345127663299992, -0.0001385320434298094,
      -0.04153911806491047, 0.04276259860299151, -0.2199478220086813, 0.0001411604247274223, 0.0001385343353064826;
  VectorXd q9_dot(kNumSamples);
  q9_dot << -0.03114140997173571, 0.1891058964085061, -0.2199472593010045, 0.0001152801664156702,
      -0.0001206088631887677, 0.1108926525713626, 0.0892519793186399, -0.219946927017527, 0.0001247277419741524,
      0.0001206108600406217;
  VectorXd q10_dot(kNumSamples);
  q10_dot << 5.421010862427522e-20, 0, 0, 0, 1.734723475976807e-18, 0, -5.421010862427522e-20, 0, 5.551115123125783e-17,
      -1.734723475976807e-18;

  const std::vector<VectorXd> q{q1, q2, q3, q4, q5, q6, q7, q8, q9, q10};
  const std::vector<VectorXd> q_dot{q1_dot, q2_dot, q3_dot, q4_dot, q5_dot, q6_dot, q7_dot, q8_dot, q9_dot, q10_dot};
  VectorXd lower_acceleration_limit(kNumJoints);
  lower_acceleration_limit << -10.50715945369842, -20.25461455189025, -154.9631704569962, -104.6612700074672,
      -277.711465225322, -10.50677759337783, -20.57044454570808, -145.6186395264131, -104.4873703219462,
      -279.2519700425768;
  VectorXd upper_acceleration_limit(kNumJoints);
  upper_acceleration_limit << 10.51273843768769, 20.25461458930144, 154.963171206302, 85.06127000746734,
      277.7114652253203, 10.5128223898978, 20.57044464133078, 145.6186354788589, 84.88737032194597, 279.2519700425739;

  VectorXd velocity_limit(kNumJoints);
  velocity_limit << 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0;

  // Compute the toppra algorithm.
  const double kStartTime = 0.0;
  const double kEndTimes = 8.64159;
  toppra::Vector times_toppra = toppra::Vector::LinSpaced(kNumSamples, kStartTime, kEndTimes);
  std::vector<double> times;
  for (size_t i = 0; i < kNumSamples; ++i) times.push_back(times_toppra(i));

  toppra::Vectors q_nominal_vectors;
  toppra::Vectors q_dot_nominal_vectors;
  for (int i = 0; i < kNumSamples; i++) {
    q_nominal_vectors.push_back(q[i]);
    q_dot_nominal_vectors.push_back(q_dot[i]);
  }

  PiecewisePolyPath hermite =
      PiecewisePolyPath::constructHermite(q_nominal_vectors, q_dot_nominal_vectors, times);

  toppra::GeometricPathPtr path;
  path = std::make_shared<toppra::PiecewisePolyPath>(hermite);

  toppra::LinearConstraintPtr velocity_constraint =
      std::make_shared<toppra::constraint::LinearJointVelocity>(-velocity_limit, velocity_limit);
  velocity_constraint->discretizationType(toppra::DiscretizationType::Interpolation);

  // Make the acceleration constraint.
  LinearConstraintPtr acceleration_constraint =
      std::make_shared<constraint::LinearJointAcceleration>(lower_acceleration_limit, upper_acceleration_limit);
  acceleration_constraint->discretizationType(DiscretizationType::Interpolation);

  // Create linear joint space constraints.
  LinearConstraintPtrs constraints{velocity_constraint, acceleration_constraint};

  // Number of gridpoints to consider
  int N = 0;
//   int N = 100;

  // Create and run toppra.
  auto algo = std::make_shared<algorithm::TOPPRA>(constraints, path);
  algo->setN(N);

  toppra::ReturnCode return_code =
      algo->computePathParametrization(0 /*Zero Initial Velocity*/, 0 /*Zero Final Velocity*/);
  ASSERT_EQ(return_code, toppra::ReturnCode::OK);

  toppra::ParametrizationData data = algo->getParameterizationData();

  // Create constant acceleration parametrizer.
  auto gridpoints = data.gridpoints;     // Grid-points used for solving the discretized problem.
  PRINT("Gridpoints has " << gridpoints.size() << " points");

  toppra::Vector vsquared = data.parametrization;  // Output parametrization (squared path velocity)
  auto spline_path = std::make_shared<parametrizer::ConstAccel>(
      path, data.gridpoints, data.parametrization);

  const toppra::Bound optimized_time_interval = spline_path->pathInterval();
  PRINT("Optimized time: " << optimized_time_interval);
  toppra::Vector time_breaks_optimized =
      toppra::Vector::LinSpaced(5 * kNumSamples, optimized_time_interval(0), optimized_time_interval(1));
  toppra::Vectors q_nominal_optimized = spline_path->eval(time_breaks_optimized, 0);
  toppra::Vectors q_dot_nominal_optimized = spline_path->eval(time_breaks_optimized, 1);
  toppra::Vectors q_ddot_nominal_optimized = spline_path->eval(time_breaks_optimized, 2);

  std::vector<VectorXd> q_nominal_optimized_mat{};
  std::vector<VectorXd> q_dot_nominal_optimized_mat{};
  std::vector<double> time_breaks_optimized_vec{};

  // Now append the rest of the optimized path points generated by toppra.
  const double kTolerance = 1.001;
  for (size_t i = 0; i < 5 * kNumSamples; ++i) {
    const double t = time_breaks_optimized[i];
    const toppra::Vector& q_t = q_nominal_optimized[i];
    const toppra::Vector& q_dot_t = q_dot_nominal_optimized[i];
    const toppra::Vector q_ddot_t = q_ddot_nominal_optimized[i];
    for (int j = 0; j < kNumJoints; j++) {
      ASSERT_LT(q_ddot_t[j], kTolerance * upper_acceleration_limit[j]);
      ASSERT_LT(q_dot_t[j], kTolerance * velocity_limit[j]);
      ASSERT_GT(q_ddot_t[j], kTolerance * lower_acceleration_limit[j]);
      ASSERT_GT(q_dot_t[j], -kTolerance * velocity_limit[j]);
    }
    q_nominal_optimized_mat.push_back(q_t);
    q_dot_nominal_optimized_mat.push_back(q_dot_t);
    time_breaks_optimized_vec.push_back(t);
  }
}
}  // namespace test

