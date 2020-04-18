#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <Eigen/Dense>

#include <Eigen/src/Core/util/Constants.h>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <chrono>

#include "toppra/geometric_path.hpp"
#include "toppra/toppra.hpp"

class ConstructPiecewisePoly : public testing::Test {
public:
  ConstructPiecewisePoly() : path(constructPath()) {

    // path has the equation: 0 * x ^ 3 + 1 * x ^ 2 + 2 x ^ 1 + 3
  }

  toppra::PiecewisePolyPath constructPath() {
    toppra::Matrix coeff{4, 2};
    coeff(0, 0) = 0;
    coeff(0, 1) = 0;
    coeff(1, 0) = 1;
    coeff(1, 1) = 1;
    coeff(2, 0) = 2;
    coeff(2, 1) = 2;
    coeff(3, 0) = 3;
    coeff(3, 1) = 3;
    toppra::Matrices coefficents = {coeff, coeff};
    toppra::PiecewisePolyPath p =
        toppra::PiecewisePolyPath(coefficents, std::vector<double>{0, 1, 2});
    return p;
  }

  toppra::PiecewisePolyPath path;
};

TEST_F(ConstructPiecewisePoly, OutputValueHasCorrectDOF) {
  toppra::Vector pos = path.eval(0.5, 0);
  ASSERT_THAT(pos.rows(), testing::Eq(2));
}

TEST_F(ConstructPiecewisePoly, CorrectOutputValueFirstSegment) {
  toppra::Vector pos = path.eval(0.5, 0);
  ASSERT_DOUBLE_EQ(pos(0), 1 * pow(0.5, 2) + 2 * pow(0.5, 1) + 3);
}

TEST_F(ConstructPiecewisePoly, CorrectOutputValueSecondSegment) {
  toppra::Vector pos = path.eval(1.5, 0);
  ASSERT_DOUBLE_EQ(pos(0), 1 * pow(0.5, 2) + 2 * pow(0.5, 1) + 3);
}

TEST_F(ConstructPiecewisePoly, ThrowWhenOutOfrange) {
  toppra::value_type out_of_range_pos{10};
  ASSERT_THROW(path.eval(out_of_range_pos, 0), std::runtime_error);
}

TEST_F(ConstructPiecewisePoly, DeriveDerivativeOfCoefficients) {
  toppra::value_type out_of_range_pos{10};
  ASSERT_THROW(path.eval(out_of_range_pos, 0), std::runtime_error);
}

TEST_F(ConstructPiecewisePoly, CorrectDerivative) {
  toppra::Vector pos = path.eval(0.5, 1);
  ASSERT_DOUBLE_EQ(pos(0), 2 * pow(0.5, 1) + 2 * 1 + 0);
}

TEST_F(ConstructPiecewisePoly, CorrectDoubldDerivative) {
  toppra::Vector pos = path.eval(0.5, 2);
  ASSERT_DOUBLE_EQ(pos(0), 2 * 1 + 0 + 0);
}

TEST_F(ConstructPiecewisePoly, ComputeManyPoints) {
  toppra::Vectors positions =
      path.eval(std::vector<toppra::value_type>{0.5, 1.2}, 0);
  ASSERT_DOUBLE_EQ(positions[0](0), 1 * pow(0.5, 2) + 2 * pow(0.5, 1) + 3);
  ASSERT_DOUBLE_EQ(positions[1](0), 1 * pow(0.2, 2) + 2 * pow(0.2, 1) + 3);
}

TEST_F(ConstructPiecewisePoly, ComputeManyPointsEigen) {
  toppra::Vector times{2};
  times << 0.5, 1.2;
  toppra::Vectors positions = path.eval(times, 0);
  ASSERT_DOUBLE_EQ(positions[0](0), 1 * pow(0.5, 2) + 2 * pow(0.5, 1) + 3);
  ASSERT_DOUBLE_EQ(positions[1](0), 1 * pow(0.2, 2) + 2 * pow(0.2, 1) + 3);
}

TEST_F(ConstructPiecewisePoly, CorrectDOF) {
  ASSERT_THAT(path.dof(), testing::Eq(2));
}

// Current profile result (Release build)
// Took ~ 400 usec to evaluate 1000 points.
// scipy.PPoly took ~ 125 usec.
// Further improvements need not yield much added benefits.
TEST(ProfileEvaluationSpeed, Test1) {
  static int dof{6};
  toppra::Matrix coeffs{4, dof};
  for (int i_col = 0; i_col < dof; i_col++) {
    coeffs.col(i_col) << 2, 4, 5, 6;
  }
  toppra::Matrices coefficients{10, coeffs};
  std::vector<toppra::value_type> path_positions;
  for (toppra::value_type s = 0; s < 10; s += 0.01) {
    path_positions.push_back(s);
  }
  std::chrono::steady_clock::time_point begin =
      std::chrono::steady_clock::now();
  toppra::PiecewisePolyPath p = toppra::PiecewisePolyPath(
      coefficients,
      std::vector<toppra::value_type>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10});

  p.eval(path_positions, 2);
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "Evaluation dones in: "
            << std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                     begin)
                   .count()
            << " usec" << std::endl;
}

class BadInputs : public testing::Test {
public:
  BadInputs() : coeff(4, 2) {
    coeff << 0, 0, 1, 1, 2, 2, 4, 4;
    coefficents.push_back(coeff);
    coefficents.push_back(coeff);
  }
  toppra::Matrix coeff;
  toppra::Matrices coefficents;
};

TEST_F(BadInputs, ThrowIfBreakPointsNotIncreasing) {
  ASSERT_THROW(
      toppra::PiecewisePolyPath(coefficents, std::vector<double>{0, 2, 1}),
      std::runtime_error);
}

TEST_F(BadInputs, ThrowIfWrongNumberOfBreakPoints) {
  ASSERT_THROW(
      toppra::PiecewisePolyPath(coefficents, std::vector<double>{0, 1, 2, 3}),
      std::runtime_error);
}
