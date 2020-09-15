#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include "gtest/gtest.h"

#include <chrono>

#include <toppra/geometric_path/piecewise_poly_path.hpp>
#include <toppra/toppra.hpp>

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
  toppra::Vector pos = path.eval_single(0.5, 0);
  ASSERT_EQ(pos.rows(), 2);
}

TEST_F(ConstructPiecewisePoly, CorrectOutputValueFirstSegment) {
  toppra::Vector pos = path.eval_single(0.5, 0);
  ASSERT_DOUBLE_EQ(pos(0), 1 * pow(0.5, 2) + 2 * pow(0.5, 1) + 3);
}

TEST_F(ConstructPiecewisePoly, CorrectOutputValueSecondSegment) {
  toppra::Vector pos = path.eval_single(1.5, 0);
  ASSERT_DOUBLE_EQ(pos(0), 1 * pow(0.5, 2) + 2 * pow(0.5, 1) + 3);
}

TEST_F(ConstructPiecewisePoly, ThrowWhenOutOfrange) {
  toppra::value_type out_of_range_pos{10};
  ASSERT_THROW(path.eval_single(out_of_range_pos, 0), std::runtime_error);
}

TEST_F(ConstructPiecewisePoly, DeriveDerivativeOfCoefficients) {
  toppra::value_type out_of_range_pos{10};
  ASSERT_THROW(path.eval_single(out_of_range_pos, 0), std::runtime_error);
}

TEST_F(ConstructPiecewisePoly, CorrectDerivative) {
  toppra::Vector pos = path.eval_single(0.5, 1);
  ASSERT_DOUBLE_EQ(pos(0), 2 * pow(0.5, 1) + 2 * 1 + 0);
}

TEST_F(ConstructPiecewisePoly, CorrectDoubldDerivative) {
  toppra::Vector pos = path.eval_single(0.5, 2);
  ASSERT_DOUBLE_EQ(pos(0), 2 * 1 + 0 + 0);
}

TEST_F(ConstructPiecewisePoly, ComputeManyPointsEigen) {
  toppra::Vector times{2};
  times << 0.5, 1.2;
  toppra::Vectors positions = path.eval(times, 0);
  ASSERT_DOUBLE_EQ(positions[0](0), 1 * pow(0.5, 2) + 2 * pow(0.5, 1) + 3);
  ASSERT_DOUBLE_EQ(positions[1](0), 1 * pow(0.2, 2) + 2 * pow(0.2, 1) + 3);
}

TEST_F(ConstructPiecewisePoly, CorrectDOF) {
  ASSERT_EQ(path.dof(), 2);
}

TEST_F(ConstructPiecewisePoly, CorrectPathInterval) {
  toppra::Bound b = path.pathInterval();
  ASSERT_DOUBLE_EQ(b[0], 0);
  ASSERT_DOUBLE_EQ(b[1], 2);
}

#ifdef TOPPRA_OPT_MSGPACK

TEST_F(ConstructPiecewisePoly, serializeInvariant) {
  std::stringstream buffer;
  path.serialize(buffer);

  toppra::PiecewisePolyPath pathNew;
  pathNew.deserialize(buffer);

  ASSERT_EQ(pathNew.dof(), 2);
  toppra::Vector v(3);
  v << 0.1, 0.2, 0.3;
  pathNew.eval(v);
}

#endif

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
  toppra::Vector path_positions{1000};
  for (size_t i = 0; i < 1000; i++) {
    path_positions(i) = std::min(0., std::max(10., (toppra::value_type)(i) / 100.0));
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


/* Python code to generate this test data

import toppra as ta

path = ta.SplineInterpolator([0, 1, 2, 3], [[0, 0], [1, 3], [2, 4], [0, 0]])

def print_cpp_code(p):
    out = ""
    for seg_idx in range(p.cspl.c.shape[1]):
        out += "coeff{:d} << ".format(seg_idx)
        for i, t in enumerate(p.cspl.c[:, seg_idx, :].flatten().tolist()):
            if i == len(p.cspl.c[:, seg_idx, :].flatten().tolist()) - 1:
                out += "{:f};\n".format(t)
            else:
                out += "{:f}, ".format(t)
    return out

print(print_cpp_code(path))
print("breakpoints: {}".format([0, 1, 2, 3]))
x_eval = [0, 0.5, 1., 1.1, 2.5]
print("Eval for x_eval = {:}\npath(x_eval)=\n{}\npath(x_eval, 1)=\n{}\npath(x_eval, 2)=\n{}".format(
    x_eval, path(x_eval), path(x_eval, 1), path(x_eval, 2)))
 */

class CompareWithScipyCubicSpline : public testing::Test {

public:
  CompareWithScipyCubicSpline() {
    toppra::Matrix coeff0{4, 2}, coeff1{4, 2}, coeff2{4, 2};
    coeff0 << -0.500000, -0.500000, 1.500000, 0.500000, 0.000000, 3.000000, 0.000000, 0.000000;
    coeff1 << -0.500000, -0.500000, 0.000000, -1.000000, 1.500000, 2.500000, 1.000000, 3.000000;
    coeff2 << -0.500000, -0.500000, -1.500000, -2.500000, 0.000000, -1.000000, 2.000000, 4.000000;
    toppra::Matrices coefficents = {coeff0, coeff1, coeff2};
    path = toppra::PiecewisePolyPath(coefficents, std::vector<double>{0, 1, 2, 3});

    x_eval.resize(5);
    x_eval << 0, 0.5, 1, 1.1, 2.5;
  // Eval for x_eval = [0, 0.5, 1.0, 1.1, 2.5]
  // path(x_eval)=
  // [[0.     0.    ]
  //  [0.3125 1.5625]
  //  [1.     3.    ]
  //  [1.1495 3.2395]
  //  [1.5625 2.8125]]
  toppra::Vector v0(2); v0 << 0, 0;
  toppra::Vector v1(2); v1 << 0.3125, 1.5625;
  toppra::Vector v2(2); v2 << 1.    , 3.    ;
  toppra::Vector v3(2); v3 << 1.1495, 3.2395;
  toppra::Vector v4(2); v4 << 1.5625, 2.8125;
  expected_pos = toppra::Vectors{v0, v1, v2, v3, v4};

  // Eval for x_eval = [0, 0.5, 1.0, 1.1, 2.5]
  // path(x_eval, 1)=
  // [[ 0.     3.   ]
  //  [ 1.125  3.125]
  //  [ 1.5    2.5  ]
  //  [ 1.485  2.285]
  //  [-1.875 -3.875]]
  v0 <<  0.   ,  3.   ;
  v1 <<  1.125,  3.125;
  v2 <<  1.5  ,  2.5  ;
  v3 <<  1.485,  2.285;
  v4 << -1.875, -3.875;
  expected_vel = toppra::Vectors{v0, v1, v2, v3, v4};

  // path(x_eval, 2)=
  // [[ 3.   1. ]
  //  [ 1.5 -0.5]
  //  [ 0.  -2. ]
  //  [-0.3 -2.3]
  //  [-4.5 -6.5]]
  v0 << 3. ,  1. ;
  v1 << 1.5, -0.5;
  v2 << 0. , -2. ;
  v3 <<-0.3, -2.3;
  v4 <<-4.5, -6.5;
  expected_acc = toppra::Vectors{v0, v1, v2, v3, v4};
  }
  

  toppra::Vector x_eval;
  toppra::PiecewisePolyPath path;
  toppra::Vectors expected_pos, expected_vel, expected_acc;
};


TEST_F(CompareWithScipyCubicSpline, 0thDerivative){

  auto res = path.eval(x_eval);
  for(int i=0; i < 5; i++){
    ASSERT_TRUE(res[i].isApprox(expected_pos[i])) << "Comparing the " << i << "th" << res[i] << expected_pos[i];
  }
}

TEST_F(CompareWithScipyCubicSpline, 1stDerivative){
  auto res = path.eval(x_eval, 1);
  for(int i=0; i < 5; i++){
    ASSERT_TRUE(res[i].isApprox(expected_vel[i])) << "Comparing the " << i << "th" << res[i] << expected_vel[i];
  }
}


TEST_F(CompareWithScipyCubicSpline, 2stDerivative){
  auto res = path.eval(x_eval, 2);
  for(int i=0; i < 5; i++){
    ASSERT_TRUE(res[i].isApprox(expected_acc[i])) << "Comparing the " << i << "th" << res[i] << "," << expected_acc[i];
  }
}

