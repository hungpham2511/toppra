#include <toppra/geometric_path.hpp>
#include <toppra/geometric_path/piecewise_poly_path.hpp>
#include <toppra/parametrizer/const_accel.hpp>
#include <toppra/toppra.hpp>

#include "gtest/gtest.h"

#define TEST_PRECISION 1e-6

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
  };
  std::shared_ptr<toppra::PiecewisePolyPath> path;
};

TEST_F(ParametrizeConstAccel, Basic) {
  toppra::Vector gridpoints = toppra::Vector::LinSpaced(10, 0, 3);
  toppra::Vector vsquared {10};
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
