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
    coeff0 << -0.500000, -0.500000, 1.500000, 0.500000, 0.000000, 3.000000, 0.000000, 0.000000;
    coeff1 << -0.500000, -0.500000, 0.000000, -1.000000, 1.500000, 2.500000, 1.000000, 3.000000;
    coeff2 << -0.500000, -0.500000, -1.500000, -2.500000, 0.000000, -1.000000, 2.000000, 4.000000;
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

TEST_F(ProblemInstance, SimpleGridpoints) {
  toppra::algorithm::TOPPRA problem{v, path};
  toppra::ReturnCode ret_code = problem.computePathParametrization();
  // ASSERT_THAT(ret_code, toppra::ReturnCode::OK) << "actual return code: " <<
  // (int)ret_code; auto problem_data = problem_ptr->getParameterizationData();
}
