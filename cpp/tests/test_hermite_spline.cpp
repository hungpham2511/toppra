#include <toppra/geometric_path/piecewise_poly_path.hpp>
#include <toppra/toppra.hpp>
#include "gtest/gtest.h"
#include "utils.hpp"

namespace toppra {

TEST(SplineHermite, BasicUsage) {
  toppra::Vectors pos = makeVectors({{0, 0}, {1, 1}, {0, 2}});
  toppra::Vectors vec = makeVectors({{1, 1}, {0, 0}, {1, 2}});
  toppra::PiecewisePolyPath path =
      toppra::PiecewisePolyPath::constructHermite(pos, vec, {0, 0.5, 1});
  auto resPos = path.eval_single(0.5);
  EXPECT_DOUBLE_EQ(resPos(0), 1);
  EXPECT_DOUBLE_EQ(resPos(1), 1);

  auto resVel = path.eval_single(0.5, 1);
  EXPECT_DOUBLE_EQ(resVel(0), 0);
  EXPECT_DOUBLE_EQ(resVel(1), 0);
}

}  // namespace toppra
