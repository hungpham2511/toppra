#include <memory>
#include <toppra/algorithm/toppra.hpp>
#include <toppra/constraint.hpp>
#include <toppra/constraint/linear_joint_acceleration.hpp>
#include <toppra/constraint/linear_joint_velocity.hpp>
#include <toppra/geometric_path.hpp>
#include <toppra/toppra.hpp>
#include "toppra/algorithm.hpp"

#include "gtest/gtest.h"

#define TOPPRA_PRECISION 1e-6

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
    v = toppra::LinearConstraintPtrs{
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
  e_K_max << 0.06666667, 0.07624309, 0.08631706, 0.09690258, 0.1005511, 0.09982804,
      0.09979021, 0.1004364, 0.10178673, 0.10184412, 0.09655088, 0.09173679, 0.08734254,
      0.08331796, 0.07962037, 0.07621325, 0.07306521, 0.07014913, 0.0674415, 0.06492188,
      0.06257244, 0.06037764, 0.05832397, 0.05639984, 0.05459563, 0.05290407,
      0.05132158, 0.04985238, 0.04852317, 0.04745694, 0.04761905, 0.05457026,
      0.06044905, 0.06527948, 0.08479263, 0.10990991, 0.13252362, 0.15269631,
      0.15777077, 0.12111776, 0.09525987, 0.07641998, 0.06232537, 0.05154506,
      0.04314353, 0.03257513, 0.02268898, 0.01495548, 0.0088349, 0.00394283, 0.;

  ASSERT_EQ(ret_code, toppra::ReturnCode::OK)
      << "actual return code: " << (int)ret_code;
  for (int i = 0; i < 51; i++)
    EXPECT_NEAR(data.controllable_sets(i, 1), e_K_max(i), TOPPRA_PRECISION)
        << "idx: " << i;
}

TEST_F(ProblemInstance, OutputParmetrization) {
  toppra::algorithm::TOPPRA problem{v, path};
  problem.setN(50);
  auto ret_code = problem.computePathParametrization();
  auto data = problem.getParameterizationData();
  toppra::Vector expected_parametrization(51);
  expected_parametrization << 0., 0.00799999, 0.01559927, 0.02295854, 0.03021812,
      0.0375065, 0.04494723, 0.05266502, 0.06079176, 0.06947278, 0.07887417, 0.08890758,
      0.08734253, 0.08331795, 0.07962036, 0.07621324, 0.0730652, 0.07014912, 0.06744149,
      0.06492187, 0.06257243, 0.06037763, 0.05832396, 0.05639983, 0.05459562,
      0.05290406, 0.05132157, 0.04985237, 0.04852316, 0.04745693, 0.04761904, 0.0285715,
      0.05376003, 0.04275653, 0.04126188, 0.04013804, 0.03912958, 0.03818766,
      0.03729606, 0.0364472, 0.03563649, 0.03486069, 0.03411724, 0.03340395, 0.03271895,
      0.03206054, 0.02268897, 0.01495547, 0.00883489, 0.00394282, 0.;

  ASSERT_EQ(ret_code, toppra::ReturnCode::OK)
      << "actual return code: " << (int)ret_code;

  for (int i = 0; i < 51; i++)
    EXPECT_NEAR(data.parametrization(i), expected_parametrization(i), TOPPRA_PRECISION)
        << "idx: " << i
        << ", abs diff=" << data.parametrization(i) - expected_parametrization(i);
}
