#include <gmock/gmock-matchers.h>
#include <memory>
#include <toppra/algorithm/toppra.hpp>
#include <toppra/constraint.hpp>
#include <toppra/constraint/linear_joint_acceleration.hpp>
#include <toppra/constraint/linear_joint_velocity.hpp>
#include <toppra/geometric_path.hpp>
#include <toppra/toppra.hpp>
#include "toppra/algorithm.hpp"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

//// Code use to generate the test scenario using the Python implementation

// import toppra as ta
// import numpy as np
//
// path = ta.SplineInterpolator([0, 1, 2, 3], [[0, 0], [1, 3], [2, 4], [0, 0]])
//
// def print_cpp_code(p):
//     out = ""
//     for seg_idx in range(p.cspl.c.shape[1]):
//         out += "coeff{:d} << ".format(seg_idx)
//         for i, t in enumerate(p.cspl.c[:, seg_idx, :].flatten().tolist()):
//             if i == len(p.cspl.c[:, seg_idx, :].flatten().tolist()) - 1:
//                 out += "{:f};\n".format(t)
//             else:
//                 out += "{:f}, ".format(t)
//     return out
//
// print(print_cpp_code(path))
// print("breakpoints: {}".format([0, 1, 2, 3]))
// x_eval = [0, 0.5, 1., 1.1, 2.5]
// print("Eval for x_eval = {:}\npath(x_eval)=\n{}\npath(x_eval, 1)=\n{}\npath(x_eval,
// 2)=\n{}".format(
//     x_eval, path(x_eval), path(x_eval, 1), path(x_eval, 2)))
//
// pc_vel = ta.constraint.JointVelocityConstraint([1.0, 1.0])
// pc_acc = ta.constraint.JointAccelerationConstraint([0.2, 0.2])
//
// instance = ta.algorithm.TOPPRA([pc_vel, pc_acc], path, gridpoints=np.linspace(0, 3,
// 51)) sdds, sds, _ = instance.compute_parameterization(0, 0)

class ProblemInstance : public testing::Test {
 public:
  ProblemInstance() {
    toppra::Matrix coeff0{4, 2}, coeff1{4, 2}, coeff2{4, 2};
    coeff0 << -0.500000, -0.500000, 1.500000, 0.500000, 0.000000, 3.000000, 0.000000,
        0.000000;
    coeff1 << -0.500000, -0.500000, 0.000000, -1.000000, 1.500000, 2.500000, 1.000000,
        3.000000;
    coeff2 << -0.500000, -0.500000, -1.500000, -2.500000, 0.000000, -1.000000, 2.000000,
        4.000000;
    toppra::Matrices coefficents = {coeff0, coeff1, coeff2};
    path = toppra::PiecewisePolyPath(coefficents, std::vector<double>{0, 1, 2, 3});
    toppra::LinearConstraintPtrs v{
        std::make_shared<toppra::constraint::LinearJointVelocity>(
            -toppra::Vector::Ones(nDof), toppra::Vector::Ones(nDof)),
        std::make_shared<toppra::constraint::LinearJointAcceleration>(
            -0.2 * toppra::Vector::Ones(nDof), 0.2 * toppra::Vector::Ones(nDof))};
  };

  toppra::PiecewisePolyPath path;
  toppra::LinearConstraintPtrs v;
  int nDof = 2;
};

TEST_F(ProblemInstance, GridpointsHasCorrectShape) {
  toppra::algorithm::TOPPRA problem{v, path};
  problem.setN(50);
  problem.computePathParametrization();
  auto data = problem.getParameterizationData();

  ASSERT_EQ(data.gridpoints.size(), 51);
  ASSERT_TRUE(data.gridpoints.isApprox(toppra::Vector::LinSpaced(51, 0, 3)));
}

TEST_F(ProblemInstance, ControllableSets) {
  toppra::algorithm::TOPPRA problem{v, path};
  problem.setN(50);
  auto ret_code = problem.computePathParametrization();
  auto data = problem.getParameterizationData();
  toppra::Vector e_K_max(51);
  e_K_max << 0.06666667, 0.07576745, 0.08538251, 0.09551183, 0.1005511, 0.09982804,
      0.09979021, 0.1004364, 0.10178673, 0.0991224, 0.09423567, 0.08976427, 0.0856601,
      0.08188198, 0.07839447, 0.07516693, 0.07217274, 0.06938868, 0.06679442, 0.0643721,
      0.06210595, 0.05998203, 0.05798797, 0.05611274, 0.05434655, 0.05268066,
      0.05110754, 0.04962257, 0.04823299, 0.04688841, 0.04761905, 0.05457026,
      0.06044905, 0.06527948, 0.075, 0.08635149, 0.09642753, 0.10534652, 0.11322472,
      0.12017269, 0.09525987, 0.07397325, 0.0568159, 0.04339803, 0.0327621, 0.02423354,
      0.01732675, 0.0116854, 0.00704361, 0.0032, 0.;
  ASSERT_EQ(ret_code, toppra::ReturnCode::OK)
      << "actual return code: " << (int)ret_code;
  ASSERT_DOUBLE_EQ(data.controllable_sets(50, 1), e_K_max(50));
  ASSERT_DOUBLE_EQ(data.controllable_sets(49, 1), e_K_max(49));
}

TEST_F(ProblemInstance, OutputParmetrization) {
  toppra::algorithm::TOPPRA problem{v, path};
  problem.setN(50);
  auto ret_code = problem.computePathParametrization();
  auto data = problem.getParameterizationData();
  toppra::Vector expected_parametrization(51);
  expected_parametrization << 0., 0.00761179, 0.01498625, 0.02225818, 0.0295301,
      0.03682582, 0.04426912, 0.05198486, 0.06010488, 0.06877432, 0.07815896,
      0.08811985, 0.08566009, 0.08188197, 0.07839446, 0.07516692, 0.07217273,
      0.06938867, 0.06679441, 0.06437209, 0.06210594, 0.05998202, 0.05798796,
      0.05611273, 0.05434654, 0.05268065, 0.05110753, 0.04962256, 0.04813771, 0.0468884,
      0.04560085, 0.04431338, 0.04320243, 0.04209148, 0.04103459, 0.04002688,
      0.03906509, 0.03814624, 0.03726759, 0.03642665, 0.03562111, 0.03484884, 0.0341079,
      0.03339645, 0.03271282, 0.02423353, 0.01732674, 0.01168539, 0.0070436, 0.00319999,
      0.;

  ASSERT_EQ(ret_code, toppra::ReturnCode::OK)
      << "actual return code: " << (int)ret_code;
  ASSERT_TRUE(data.parametrization.isApprox(expected_parametrization))
      << "actual parametrization: \n " << data.parametrization;
}
