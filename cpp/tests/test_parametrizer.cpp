#include <toppra/geometric_path.hpp>
#include <toppra/geometric_path/piecewise_poly_path.hpp>
#include <toppra/parametrizer/const_accel.hpp>
#include <toppra/toppra.hpp>

#include "gtest/gtest.h"

#define TEST_PRECISION 1e-3

class ParametrizeConstAccel : public testing::Test {
 public:
  ParametrizeConstAccel() {
    toppra::Matrix coeff0{4, 2}, coeff1{4, 2}, coeff2{4, 2};
    coeff0 << -0.500000, -0.500000, 1.500000, 0.500000, 0.000000, 3.000000, 0.000000,
        0.000000;
    coeff1 << -0.500000, -0.500000, 0.000000, -1.000000, 1.500000, 2.500000, 1.000000,
        3.000000;
    coeff2 << -0.500000, -0.500000, -1.500000, -2.500000, 0.000000, -1.000000, 2.000000,
        4.000000;
    toppra::Matrices coefficents = {coeff0, coeff1, coeff2};
    path = std::make_shared<toppra::PiecewisePolyPath>(coefficents,
                                                       std::vector<double>{0, 1, 2, 3});

    TOPPRA_LOG_DEBUG("coeff0: " << coeff0);
    TOPPRA_LOG_DEBUG("coeff1: " << coeff1);
    TOPPRA_LOG_DEBUG("coeff2: " << coeff2);
  };
  std::shared_ptr<toppra::PiecewisePolyPath> path;
};

TEST_F(ParametrizeConstAccel, Basic) {
  toppra::Vector gridpoints = toppra::Vector::LinSpaced(10, 0, 3);
  toppra::Vector vsquared{10};
  vsquared << 0, 0.1, 0.2, 0.3, 0.5, 0.5, 0.3, 0.2, 0.1, 0.0;
  auto p = toppra::parametrizer::ConstAccel(path, gridpoints, vsquared);

  // bound
  auto bound = p.pathInterval();
  ASSERT_EQ(bound[0], 0);
  ASSERT_GE(bound[1], 0);

  // evaluation
  toppra::Vector ts = toppra::Vector::LinSpaced(10, bound[0], bound[1]);
  auto qs = p.eval(ts);
  auto qds = p.eval(ts, 1);
  auto qdds = p.eval(ts, 2);
  ASSERT_TRUE(p.validate());
  ASSERT_EQ(qds[0][0], 0);
  ASSERT_EQ(qds[0][1], 0);

  ASSERT_EQ(qds[9][0], 0);
  ASSERT_EQ(qds[9][1], 0);
}

TEST_F(ParametrizeConstAccel, Correctness) {
  toppra::Vector gridpoints = toppra::Vector::LinSpaced(10, 0, 3);
  TOPPRA_LOG_DEBUG("gridpoints " << gridpoints);
  toppra::Vector vsquared{10};
  vsquared << 0, 0.1, 0.2, 0.3, 0.5, 0.5, 0.3, 0.2, 0.1, 0.0;
  auto p = toppra::parametrizer::ConstAccel(path, gridpoints, vsquared);
  TOPPRA_LOG_DEBUG("Duration: " << p.pathInterval());

  // evaluation
  toppra::Vector ts = toppra::Vector{4};
  ts << 1, 0, 3, 8;
  auto qs = p.eval(ts);
  auto qds = p.eval(ts, 1);
  auto qdds = p.eval(ts, 2);
  toppra::Vector q0, qd0, qdd0;

  // test for t = 0
  {
    // t = 0
    toppra::value_type t = 0, u = 0.15, v = 0, s = 0;
    q0 = path->eval_single(s, 0);
    qd0 = path->eval_single(s, 1) * v;
    qdd0 = path->eval_single(s, 1) * u + path->eval_single(s, 2) * v * v;
    TOPPRA_LOG_DEBUG(qdd0);

    for (int i = 0; i < 2; i++) {
      ASSERT_FLOAT_EQ(qs[1][i], q0[i]);
      ASSERT_FLOAT_EQ(qds[1][i], qd0[i]);
      ASSERT_FLOAT_EQ(qdds[1][i], qdd0[i]);
    }
  }

  {
    // t = 3
    toppra::value_type t = 3, u = 0.15, v = 0.447214 + (3 - 2.98142) * 0.15,
                       s = 0.6666666 + 0.447214 * (3 - 2.98142) +
                           0.5 * 0.15 * (3 - 2.98142) * (3 - 2.98142);
    q0 = path->eval_single(s, 0);
    qd0 = path->eval_single(s, 1) * v;
    qdd0 = path->eval_single(s, 1) * u + path->eval_single(s, 2) * v * v;

    for (int i = 0; i < 2; i++) {
      ASSERT_NEAR(qs[2][i], q0[i], TEST_PRECISION);
      ASSERT_NEAR(qds[2][i], qd0[i], TEST_PRECISION);
      ASSERT_NEAR(qdds[2][i], qdd0[i], TEST_PRECISION);
    }
  }
}

TEST(ParametrizeConstAccelNoFixture, BoundsViolation) {
    toppra::Vectors positions = {
      toppra::Vector::Zero(1),
      toppra::Vector::Zero(1)
    };
    toppra::Vectors velocities = {
      toppra::Vector::Zero(1),
      toppra::Vector::Zero(1)
    };

    double PATH_LEN = 3.51363644474459757560680;

    toppra::Vector times = toppra::Vector::LinSpaced(2, 0, PATH_LEN);
    auto std_times = std::vector<double>(times.data(), times.data() + times.size());

    auto path = toppra::PiecewisePolyPath::CubicHermiteSpline(positions, velocities, std_times);
    auto ppath = std::make_shared<toppra::PiecewisePolyPath>(path);

    toppra::Vector gridpoints = toppra::Vector::LinSpaced(10, 0, PATH_LEN);
    toppra::Vector vsquared{10};
    vsquared << 0, 0.1, 0.2, 0.3, 0.5, 0.5, 0.3, 0.2, 0.1, 0.0;
    auto p = toppra::parametrizer::ConstAccel(ppath, gridpoints, vsquared);

    auto bound = p.pathInterval();

    // Assert different ways that the evaluation attempt can fail
    const double EPS = 1.5e-8;
    const double REL_EPS = 1e-6;
    EXPECT_NO_THROW(p.eval_single(bound[0]));
    EXPECT_NO_THROW(p.eval_single(bound[0] - EPS / 2));
    EXPECT_THROW(p.eval_single(bound[0] - EPS), std::runtime_error);
    EXPECT_NO_THROW(p.eval_single(bound[1]));
    EXPECT_THROW(p.eval_single(bound[1] * (1 + REL_EPS) + 2 * EPS), std::runtime_error);
}
