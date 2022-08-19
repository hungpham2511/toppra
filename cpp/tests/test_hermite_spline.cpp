#include <toppra/geometric_path/piecewise_poly_path.hpp>
#include <toppra/toppra.hpp>
#include "gtest/gtest.h"
#include "utils.hpp"

namespace toppra {

TEST(SplineHermite, BasicUsage) {
  Vectors pos = makeVectors({{0, 0}, {1, 1}, {0, 2}});
  Vectors vec = makeVectors({{1, 1}, {0, 0}, {1, 2}});
  PiecewisePolyPath path =
      PiecewisePolyPath::constructHermite(pos, vec, {0, 0.5, 1});
  auto resPos = path.eval_single(0.5);
  EXPECT_DOUBLE_EQ(resPos(0), 1);
  EXPECT_DOUBLE_EQ(resPos(1), 1);

  auto resVel = path.eval_single(0.5, 1);
  EXPECT_DOUBLE_EQ(resVel(0), 0);
  EXPECT_DOUBLE_EQ(resVel(1), 0);

  Vector X = Vector::LinSpaced(10, 0, 1);
  PRINT(X.transpose());
  auto Y = path.eval(X);

  Matrix Ye;
  Ye.resize(Y.size(), 2);
  for (int i=0; i < Y.size(); i++) Ye.row(i) = Y.at(i);

  PRINT(Ye.col(0).transpose());
  PRINT(Ye.col(1).transpose());

  // calculate using scipy
  Vector Z(10);
  Z << 0. , 0.19341564 , 0.48559671 , 0.77777778 , 0.97119342 , 0.96021948 , 0.7037037 , 0.3484225 , 0.05898491 , 0.;
  ASSERT_TRUE((Ye.col(0) - Z).cwiseAbs().maxCoeff() < 0.001);
}


TEST(SplineHermite, ProposeGridpoints) {
  Vectors pos = makeVectors({{0, 0}, {1, 1}, {0, 2}});
  Vectors vec = makeVectors({{1, 1}, {0, 0}, {1, 2}});
  PiecewisePolyPath path =
      PiecewisePolyPath::constructHermite(pos, vec, {0, 0.5, 1});

  auto gridpoint = path.proposeGridpoints();

  // Basic assertion
  EXPECT_TRUE(gridpoint.size() > 0);
  PRINT(gridpoint);

  // Gridpoints must be increasing
  auto N = gridpoint.size();
  Vector gridpoint_diff {N - 1};
  gridpoint_diff = gridpoint.tail(N - 1) - gridpoint.head(N - 1);
  EXPECT_TRUE(gridpoint_diff.minCoeff() > 0);
  
}


}  // namespace toppra
