#include <memory>
#include <toppra/algorithm.hpp>
#include <toppra/algorithm/toppra.hpp>
#include <toppra/constraint.hpp>
#include <toppra/constraint/linear_joint_acceleration.hpp>
#include <toppra/constraint/linear_joint_velocity.hpp>
#include <toppra/geometric_path.hpp>
#include <toppra/geometric_path/piecewise_poly_path.hpp>
#include <toppra/parametrizer/const_accel.hpp>
#ifdef BUILD_WITH_qpOASES
#include <toppra/solver/qpOASES-wrapper.hpp>
#endif
#ifdef BUILD_WITH_GLPK
#include <toppra/solver/glpk-wrapper.hpp>
#endif
#include <toppra/solver/seidel.hpp>
#include <toppra/toppra.hpp>

#include "gtest/gtest.h"

#define TEST_PRECISION 1e-6

// clang-format off

/* Code use to generate the test scenario using the Python implementation
// Diable format to keep python code identation

import toppra as ta
import numpy as np

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

pc_vel = ta.constraint.JointVelocityConstraint([1.0, 1.0])
pc_acc = ta.constraint.JointAccelerationConstraint([0.2, 0.2], discretization_scheme=0)

instance = ta.algorithm.TOPPRA([pc_vel, pc_acc], path, gridpoints=np.linspace(0, 3, 51), solver_wrapper='qpoases')
sdds, sds, _, K = instance.compute_parameterization(0, 0, return_data=True)
feasible_sets = instance.compute_feasible_sets().

 */
// clang-format on

template<typename Solver>
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
    path = std::make_shared<toppra::PiecewisePolyPath>(coefficents, std::vector<double>{0, 1, 2, 3});
    v = toppra::LinearConstraintPtrs{
        std::make_shared<toppra::constraint::LinearJointVelocity>(
            -toppra::Vector::Ones(nDof), toppra::Vector::Ones(nDof)),
        std::make_shared<toppra::constraint::LinearJointAcceleration>(
            -0.2 * toppra::Vector::Ones(nDof), 0.2 * toppra::Vector::Ones(nDof))};
  };

  std::shared_ptr<toppra::PiecewisePolyPath> path;
  toppra::LinearConstraintPtrs v;
  int nDof = 2;
};

using SolverTypes = ::testing::Types<
#ifdef BUILD_WITH_qpOASES
  toppra::solver::qpOASESWrapper,
#endif
#ifdef BUILD_WITH_GLPK
  // toppra::solver::GLPKWrapper, // Failure due to numerical issue.
#endif
  toppra::solver::Seidel>;
TYPED_TEST_SUITE(ProblemInstance, SolverTypes);

TYPED_TEST(ProblemInstance, ControllableSets) {
  toppra::algorithm::TOPPRA problem{this->v, this->path};
  problem.setN(50);
  problem.solver(std::make_shared<TypeParam>());
  auto ret_code = problem.computePathParametrization();
  const auto& data = problem.getParameterizationData();

  ASSERT_EQ(data.gridpoints.size(), 51);
  ASSERT_TRUE(data.gridpoints.isApprox(toppra::Vector::LinSpaced(51, 0, 3)));

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
    EXPECT_NEAR(data.controllable_sets(i, 1), e_K_max(i), TEST_PRECISION)
        << "idx: " << i;
}

TYPED_TEST(ProblemInstance, OutputParmetrization) {
  toppra::algorithm::TOPPRA problem{this->v, this->path};
  problem.setN(50);
  problem.solver(std::make_shared<TypeParam>());
  auto ret_code = problem.computePathParametrization();
  const auto& data = problem.getParameterizationData();
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
    EXPECT_NEAR(data.parametrization(i), expected_parametrization(i), TEST_PRECISION)
        << "idx: " << i
        << ", abs diff=" << data.parametrization(i) - expected_parametrization(i);

  // First and last must be zero.
  EXPECT_DOUBLE_EQ(data.parametrization(0), 0);
  EXPECT_DOUBLE_EQ(data.parametrization(50), 0);
}

TYPED_TEST(ProblemInstance, FeasibleSets) {
  toppra::algorithm::TOPPRA problem{this->v, this->path};
  problem.setN(50);
  problem.solver(std::make_shared<TypeParam>());
  auto ret_code = problem.computeFeasibleSets();
  const auto& data = problem.getParameterizationData();
  toppra::Vector expected_feasible_max(51);
  expected_feasible_max << 0.06666667, 0.07624309, 0.08631706, 0.09690258, 0.1005511,
      0.09982804, 0.09979021, 0.1004364, 0.10178673, 0.10388394, 0.10679654, 0.11062383,
      0.11550389, 0.12162517, 0.12924407, 0.13871115, 0.15051124, 0.16532619,
      0.18413615, 0.20838854, 0.24029219, 0.27052997, 0.2601227, 0.2447933, 0.22462845,
      0.2, 0.17154989, 0.14013605, 0.10674847, 0.07241209, 0.04761905, 0.05457026,
      0.06044905, 0.06527948, 0.08479263, 0.10990991, 0.13252362, 0.15269631,
      0.15777077, 0.12111776, 0.09525987, 0.07641998, 0.06232537, 0.05154506,
      0.04314353, 0.03648939, 0.0311448, 0.02679888, 0.02322632, 0.02026086, 0.01777778;

  ASSERT_EQ(ret_code, toppra::ReturnCode::OK)
      << "actual return code: " << (int)ret_code;

  for (int i = 0; i < 51; i++)
    EXPECT_NEAR(data.feasible_sets(i, 1), expected_feasible_max(i), TEST_PRECISION)
        << "idx: " << i
        << ", abs diff=" << data.parametrization(i) - expected_feasible_max(i);
}

TYPED_TEST(ProblemInstance, ParametrizeOutputTrajectory) {
  toppra::algorithm::TOPPRA problem{this->v, this->path};
  problem.setN(50);
  problem.solver(std::make_shared<TypeParam>());
  auto ret_code = problem.computePathParametrization();

  TOPPRA_LOG_DEBUG("Pre constructed");
  toppra::parametrizer::ConstAccel output_traj{
      this->path, problem.getParameterizationData().gridpoints,
      problem.getParameterizationData().parametrization};

  // Qualitative assertion
  ASSERT_TRUE(output_traj.validate());
  ASSERT_EQ(output_traj.pathInterval()[0], 0);
  ASSERT_GE(output_traj.pathInterval()[1], 0);
  auto interval = output_traj.pathInterval();

  // Qualitative assertion
  for (int i = 0; i < this->path->dof(); i++) {
    ASSERT_EQ(output_traj.eval_single(0)[i], this->path->eval_single(0)[i]);
    ASSERT_EQ(output_traj.eval_single(output_traj.pathInterval()[1])[i],
              this->path->eval_single(this->path->pathInterval()[1])[i]);
  }
}
