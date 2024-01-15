#include "test_cubic_spline_gendata_out.hpp"
#include "utils.hpp"
#include "gtest/gtest.h"
#include <algorithm>
#include <array>
#include <iostream>
#include <sstream>
#include <toppra/geometric_path/piecewise_poly_path.hpp>
#include <toppra/toppra.hpp>
#include <utility>

namespace toppra {

void EXPECT_VECTORS_EQ(Vectors const &v1, Vectors const &v2) {
  EXPECT_EQ(v1.size(), v2.size())
      << "Wrong size: " << v1.size() << " != " << v2.size();
  for (int i = 0; i < v1.size(); i++) {
    auto diff = v1.at(i) - v2.at(i);
    EXPECT_LE(diff.cwiseAbs().maxCoeff(), 0.001)
        << v1.at(i).transpose() << " " << v2.at(i).transpose();
  }
}

BoundaryCond makeBoundaryCond(const int order,
                              const std::vector<value_type> &values) {
  BoundaryCond cond;
  cond.order = order;
  cond.values.resize(values.size());
  for (std::size_t i = 0; i < values.size(); i++)
    cond.values(i) = values[i];
  return cond;
}

class CubicSpline : public testing::Test {
protected:
  CubicSpline() {}
  virtual ~CubicSpline() {}
  static void SetUpTestSuite() {
    positions = makeVectors({{1.3, 2.1, 4.35, 2.14, -7.31, 4.31},
                             {1.5, -4.3, 1.23, -4.3, 2.13, 6.24},
                             {-3.78, 1.53, 8.12, 12.75, 9.11, 5.42},
                             {6.25, 8.12, 9.52, 20.42, 5.21, 8.31},
                             {7.31, 3.53, 8.41, 9.56, -3.15, 4.83}});
    times.resize(5);
    times << 0.9, 1.3, 2.2, 2.4, 2.6;
  }

  void ConstructCubicSpline(const Vectors &positions, const Vector &times,
                            const BoundaryCondFull &bc_type) {

    path = toppra::PiecewisePolyPath::CubicSpline(positions, times, bc_type);
  }

  void AssertSplineKnots() {
    Vectors actual_positions = path.eval(times, 0);
    for (size_t i = 0; i < actual_positions.size(); i++) {
      ASSERT_TRUE(positions[i].isApprox(actual_positions[i]));
    }
  }

  void AssertSplineBoundaryConditions(const BoundaryCondFull &bc_type) {
    ASSERT_TRUE(
        (path.eval_single(times(0), bc_type[0].order) - bc_type[0].values)
            .norm() < TOPPRA_NEARLY_ZERO);
    ASSERT_TRUE((path.eval_single(times(times.rows() - 1), bc_type[1].order) -
                 bc_type[1].values)
                    .norm() < TOPPRA_NEARLY_ZERO);
  }

  toppra::PiecewisePolyPath path;
  static Vectors positions;
  static Vector times;
};

Vectors CubicSpline::positions;
Vector CubicSpline::times;

TEST_F(CubicSpline, ClampedSpline) {
  BoundaryCond bc{1, std::vector<value_type>{0, 0, 0, 0, 0, 0}};
  BoundaryCondFull bc_type{bc, bc};
  ConstructCubicSpline(positions, times, bc_type);
  AssertSplineKnots();
  AssertSplineBoundaryConditions(bc_type);
}

TEST_F(CubicSpline, ClampedSplineString) {
  BoundaryCond bc{"clamped"};
  BoundaryCondFull bc_type{bc, bc};
  ConstructCubicSpline(positions, times, bc_type);
  AssertSplineKnots();
  // AssertSplineBoundaryConditions(bc_type);
}

TEST_F(CubicSpline, NaturalSpline) {
  BoundaryCond bc{2, std::vector<value_type>{0, 0, 0, 0, 0, 0}};
  BoundaryCondFull bc_type{bc, bc};
  ConstructCubicSpline(positions, times, bc_type);
  AssertSplineKnots();
  AssertSplineBoundaryConditions(bc_type);
}

TEST_F(CubicSpline, FirstOrderBoundaryConditions) {
  BoundaryCondFull bc_type{BoundaryCond{1, std::vector<value_type>{1.25, 0, 4.12, 1.75, 7.43, 5.31}},
                           BoundaryCond{1, std::vector<value_type>{3.51, 5.32, 4.63, 0, -3.12, 3.53}}};
  ConstructCubicSpline(positions, times, bc_type);
  AssertSplineKnots();
  AssertSplineBoundaryConditions(bc_type);
}

TEST_F(CubicSpline, SecondOrderBoundaryConditions) {
  BoundaryCondFull bc_type{
      BoundaryCond{2, std::vector<value_type>{1.52, -4.21, 7.21, 9.31, -1.53, 7.54}},
      BoundaryCond{2, std::vector<value_type>{-5.12, 8.21, 9.12, 5.12, 24.12, 9.42}}};
  ConstructCubicSpline(positions, times, bc_type);
  AssertSplineKnots();
  AssertSplineBoundaryConditions(bc_type);
}

TEST_F(CubicSpline, FirstOrderAndSecondOrderBoundaryConditions) {
  BoundaryCondFull bc_type{
      BoundaryCond{1, std::vector<value_type>{1.52, -4.21, 7.21, 9.31, -1.53, 7.54}},
      BoundaryCond{2, std::vector<value_type>{-5.12, 8.21, 9.12, 5.12, 24.12, 9.42}}};
  ConstructCubicSpline(positions, times, bc_type);
  AssertSplineKnots();
  AssertSplineBoundaryConditions(bc_type);
}

TEST_F(CubicSpline, BadPositions) {
  Vectors bad_positions = makeVectors({{1.3, 2.1, 4.35, 2.14, -7.31, 4.31},
                                       {1.5, -4.3, 1.23, -4.3, 2.13, 6.24},
                                       {-3.78, 1.53, 8.12, 12.75, 9.11, 5.42},
                                       {6.25, 8.12, 9.52, 5.21, 8.31},
                                       {7.31, 3.53, 8.41, 9.56, -3.15, 4.83}});
  BoundaryCondFull bc_type{
      BoundaryCond{2, std::vector<value_type>{1.52, -4.21, 7.21, 9.31, -1.53, 7.54}},
      BoundaryCond{2, std::vector<value_type>{-5.12, 8.21, 9.12, 5.12, 24.12, 9.42}}};
  ASSERT_THROW(ConstructCubicSpline(bad_positions, times, bc_type),
               std::runtime_error);

  bad_positions = makeVectors({{1.3, 2.1, 4.35, 2.14, -7.31, 4.31},
                               {1.5, -4.3, 1.23, -4.3, 2.13, 6.24},
                               {-3.78, 1.53, 8.12, 12.75, 9.11, 5.42},
                               {6.25, 8.12, 11.23, 9.52, 5.21, 8.31}});
  ASSERT_THROW(ConstructCubicSpline(bad_positions, times, bc_type),
               std::runtime_error);
}

TEST_F(CubicSpline, BadTimes) {
  BoundaryCondFull bc_type{
      BoundaryCond{2, std::vector<value_type>{1.52, -4.21, 7.21, 9.31, -1.53, 7.54}},
      BoundaryCond{2, std::vector<value_type>{-5.12, 8.21, 9.12, 5.12, 24.12, 9.42}}};
  Vector bad_times(1);
  bad_times << 1;
  ASSERT_THROW(ConstructCubicSpline(positions, bad_times, bc_type),
               std::runtime_error);

  bad_times.resize(4);
  bad_times << 0.11, 1.23, 4.35, 6.75;
  ASSERT_THROW(ConstructCubicSpline(positions, bad_times, bc_type),
               std::runtime_error);

  bad_times.resize(5);
  bad_times << 0.11, 1.23, 4.35, 6.75, 1.23;
  ASSERT_THROW(ConstructCubicSpline(positions, bad_times, bc_type),
               std::runtime_error);
}

TEST_F(CubicSpline, BadBoundaryConditions) {
  BoundaryCondFull bc_type{
      BoundaryCond{2, std::vector<value_type>{1.52, 7.21, 9.31, -1.53, 7.54}},
      BoundaryCond{2, std::vector<value_type>{-5.12, 8.21, 9.12, 5.12, 24.12, 9.42}}};
  ASSERT_THROW(ConstructCubicSpline(positions, times, bc_type),
               std::runtime_error);

  bc_type = {BoundaryCond{2, std::vector<value_type>{1.52, 7.21, 9.31}},
             BoundaryCond{2, std::vector<value_type>{-5.12, 8.21, 9.12}}};
  ASSERT_THROW(ConstructCubicSpline(positions, times, bc_type),
               std::runtime_error);

  bc_type = {BoundaryCond{3, std::vector<value_type>{1.52, 7.21, 9.31, 2.52, 4.41, 5.54}},
             BoundaryCond{3, std::vector<value_type>{-5.12, 8.21, 9.12, -1.32, 3.53, 9.21}}};
  ASSERT_THROW(ConstructCubicSpline(positions, times, bc_type),
               std::runtime_error);
}

class CubicSplineCompScipy : public testing::Test {
protected:
  CubicSplineCompScipy() {}
  virtual ~CubicSplineCompScipy() {}

  // Initialize the test suite with the given string
  void parse_1d(std::istringstream &is, Vector &times) {
    int N;
    is >> N; // throwaway
    is >> N;
    times.resize(N);
    for (int i = 0; i < N; i++) {
      is >> times(i);
    }
    //   std::cout << times << std::endl;
  }

  void parse_2d(std::istringstream &is, Vectors &positions) {
    int _x, N, d;
    is >> _x;
    is >> N >> d;
    positions.resize(N);
    for (int i = 0; i < N; i++) {
      positions.at(i).resize(d);
      for (int j = 0; j < d; j++) {
        is >> positions.at(i)(j);
      }
      // std::cout << positions.at(i) << std::endl;
    }
  }

  BoundaryCondFull parse_bc(std::istringstream & is){
    std::string bc_type;
    is >> bc_type;
    BoundaryCond bc{bc_type};

    is >> bc_type;
    BoundaryCond bc1{bc_type};
    BoundaryCondFull bc_full{bc, bc1};
    return bc_full;
  }

  void initialize(const char *str) {
    std::istringstream is(str);
    parse_1d(is, times);
    parse_2d(is, positions);
    auto bc = parse_bc(is);
    path = toppra::PiecewisePolyPath::CubicSpline(positions, times, bc);

    parse_1d(is, times_test);
    parse_2d(is, positions_test);
    parse_2d(is, velocities_test);
    parse_2d(is, acceleration_test);
  }

  void test() {
    auto positions_eval = path.eval(times_test);
    auto velocities_eval = path.eval(times_test, 1);
    auto accelerations_eval = path.eval(times_test, 2);
    EXPECT_VECTORS_EQ(positions_eval, positions_test);
    EXPECT_VECTORS_EQ(velocities_eval, velocities_test);
    EXPECT_VECTORS_EQ(accelerations_eval, acceleration_test);
  }

  toppra::PiecewisePolyPath path;
  Vectors positions, positions_test, velocities_test, acceleration_test;
  Vector times, times_test;
};

TEST_F(CubicSplineCompScipy, Test1) {
  initialize(_toppra_cubic_spline_test1);
  test();
}

TEST_F(CubicSplineCompScipy, Test2) {
  initialize(_toppra_cubic_spline_test2);
  test();
}

TEST_F(CubicSplineCompScipy, Test3) {
  initialize(_toppra_cubic_spline_test3);
  test();
}

} // namespace toppra
