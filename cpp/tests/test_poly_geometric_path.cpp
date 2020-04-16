#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include <stdexcept>

#include "toppra/geometric_path.hpp"
#include "toppra/toppra.hpp"

class ConstructPiecewisePoly : public testing::Test {
  public:
  ConstructPiecewisePoly(): path(constructPath()) {
 
    // path has the equation: 0 * x ^ 3 + 1 * x ^ 2 + 2 x ^ 1 + 3
  }

  toppra::PiecewisePolyPath constructPath(){
   toppra::Matrix coeff {4, 2};
    coeff(0, 0) = 0; coeff(0, 1) = 0;
    coeff(1, 0) = 1; coeff(1, 1) = 1;
    coeff(2, 0) = 2; coeff(2, 1) = 2;
    coeff(3, 0) = 3; coeff(3, 1) = 3;
    toppra::Matrices coefficents = {
      coeff, coeff
    };
    toppra::PiecewisePolyPath p = toppra::PiecewisePolyPath(coefficents, std::vector<double> {0, 1, 2});
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



// throw when path_positions not increasing
// throw when pos is not path_positions
// throw when break_points have wrong size
