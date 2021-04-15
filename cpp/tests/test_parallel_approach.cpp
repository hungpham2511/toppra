// Author: Niels Dehio, 13. September 2020
// Author: JS00000. March 2021
#include <toppra/algorithm.hpp>
#include <toppra/algorithm/toppra.hpp>
#include <toppra/constraint/linear_joint_acceleration.hpp>
#include <toppra/constraint/linear_joint_velocity.hpp>
#include <toppra/geometric_path/piecewise_poly_path.hpp>
#include <toppra/parametrizer/const_accel.hpp>
#include <toppra/toppra.hpp>
#include <toppra/solver/qpOASES-wrapper.hpp>
#include <toppra/solver/seidel.hpp>
#include <toppra/solver/seidel-parallel.hpp>

#include "utils.hpp"
#include "gtest/gtest.h"
#include <iostream>

#define RTOL 0.01
#define ATOL 0.0

toppra::BoundaryCond makeBoundaryCond(const int order, const std::vector<toppra::value_type> &values) {
    toppra::BoundaryCond cond;
    cond.order = order;
    cond.values.resize(values.size());
    for (std::size_t i = 0; i < values.size(); i++) cond.values(i) = values[i];
    return cond;
}

// This cpp-example implements the python-example described here:
// https://hungpham2511.github.io/toppra/quickstart.html
class ParallelApproach : public testing::Test {
 public:

  const int numJoints = 6;
  const int gridN = 1000;
  toppra::GeometricPathPtr path;
  toppra::LinearConstraintPtrs constraints;
  toppra::Vector velLimitLower;
  toppra::Vector velLimitUpper;
  toppra::Vector accLimitLower;
  toppra::Vector accLimitUpper;

  ParallelApproach(){

    // //#### create piecewise polynomial geometric path ####
    // toppra::Vector position0{numJoints}, position1{numJoints}, position2{numJoints};
    // position0 << 0.0, 0.0;
    // position1 << 1.0, 2.0;
    // position2 << 2.0, 0.0;
    // toppra::Vector velocity0{numJoints}, velocity1{numJoints}, velocity2{numJoints};
    // velocity0 << 0.0, 0.0;
    // velocity1 << 1.0, 1.0;
    // velocity2 << 0.0, 0.0;
    // toppra::Vectors positions = {position0, position1,
    //                              position2};  //[(0, 0), (1, 2), (2, 0)]
    // toppra::Vectors velocities = {velocity0, velocity1,
    //                               velocity2};  //[(0, 0), (1, 1), (0, 0)]

    // std::vector<toppra::value_type> steps;
    // steps = std::vector<toppra::value_type>{0, 1, 2};
    // toppra::PiecewisePolyPath hermite =
    //     toppra::PiecewisePolyPath::constructHermite(positions, velocities, steps);
    // path = std::make_shared<toppra::PiecewisePolyPath>(hermite);

    // //#### create linear joint-space constraints ####
    // velLimitLower = Eigen::VectorXd::Zero(numJoints);
    // velLimitUpper = Eigen::VectorXd::Zero(numJoints);
    // accLimitLower = Eigen::VectorXd::Zero(numJoints);
    // accLimitUpper = Eigen::VectorXd::Zero(numJoints);
    // velLimitLower << -1, -0.5;
    // velLimitUpper << 1, 0.5;
    // accLimitLower << -0.05, -0.1;
    // accLimitUpper << 0.2, 0.3;

    // toppra::LinearConstraintPtr ljv, lja;
    // ljv = std::make_shared<toppra::constraint::LinearJointVelocity>(
    //     velLimitLower, velLimitUpper);  //[[-1, 1], [-0.5, 0.5]]
    // lja = std::make_shared<toppra::constraint::LinearJointAcceleration>(
    //     accLimitLower, accLimitUpper);  //[[-0.05, 0.2], [-0.1, 0.3]]
    // lja->discretizationType(toppra::DiscretizationType::Interpolation);
    // constraints = toppra::LinearConstraintPtrs{ljv, lja};




    // // #### create piecewise polynomial geometric path ####
    // toppra::Vector position0{numJoints}, position1{numJoints}, position2{numJoints};
    // position0 << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    // position1 << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;
    // position2 << 2.0, 0.0, 1.0, 0.0, 3.0, 7.0;
    // toppra::Vector velocity0{numJoints}, velocity1{numJoints}, velocity2{numJoints};
    // velocity0 << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    // velocity1 << 1.0, 1.0, 1.0, 1.0, 1.0, 1.0;
    // velocity2 << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    // toppra::Vectors positions = {position0, position1,
    //                              position2};  //[(0, 0), (1, 2), (2, 0)]
    // toppra::Vectors velocities = {velocity0, velocity1,
    //                               velocity2};  //[(0, 0), (1, 1), (0, 0)]

    // std::vector<toppra::value_type> steps;
    // steps = std::vector<toppra::value_type>{0, 1, 2};
    // toppra::PiecewisePolyPath hermite =
    //     toppra::PiecewisePolyPath::constructHermite(positions, velocities, steps);
    // path = std::make_shared<toppra::PiecewisePolyPath>(hermite);



    toppra::Vectors positions;
    toppra::Vector times;
    positions = toppra::makeVectors({{1.3, 2.1, 4.35, 2.14, -7.31, 4.31},
                                     {1.5, -4.3, 1.23, -4.3, 2.13, 6.24},
                                     {-3.78, 1.53, 8.12, 12.75, 9.11, 5.42},
                                     {6.25, 8.12, 9.52, 20.42, 5.21, 8.31},
                                     {7.31, 3.53, 8.41, 9.56, -3.15, 4.83}});
    times.resize (5);
    times << 0, 1, 2, 3, 4;
    toppra::BoundaryCond bc = makeBoundaryCond(1, {0, 0, 0, 0, 0, 0});
    std::array<toppra::BoundaryCond, 2> bc_type {bc, bc};
    path = std::make_shared<toppra::PiecewisePolyPath>(positions, times, bc_type);

    //#### create linear joint-space constraints ####
    velLimitLower = Eigen::VectorXd::Zero(numJoints);
    velLimitUpper = Eigen::VectorXd::Zero(numJoints);
    accLimitLower = Eigen::VectorXd::Zero(numJoints);
    accLimitUpper = Eigen::VectorXd::Zero(numJoints);
    velLimitLower << -1, -0.5, -1, -1, -1, -1;
    velLimitUpper << 1, 0.5, 1, 1, 1, 1;
    accLimitLower << -0.05, -0.1, -0.1, -0.1, -0.1, -0.1;
    accLimitUpper << 0.2, 0.3, 0.3, 0.3, 0.3, 0.3;

    toppra::LinearConstraintPtr ljv, lja;
    ljv = std::make_shared<toppra::constraint::LinearJointVelocity>(
        velLimitLower, velLimitUpper);  //[[-1, 1], [-0.5, 0.5]]
    lja = std::make_shared<toppra::constraint::LinearJointAcceleration>(
        accLimitLower, accLimitUpper);  //[[-0.05, 0.2], [-0.1, 0.3]]
    lja->discretizationType(toppra::DiscretizationType::Interpolation);
    constraints = toppra::LinearConstraintPtrs{ljv, lja};

  };

  void formatVecToMat(const std::vector<Eigen::VectorXd,
                                        Eigen::aligned_allocator<Eigen::VectorXd>>& vec,
                      Eigen::MatrixXd& mat) {
    mat.resize(vec.at(0).rows(), vec.size());
    for (size_t i = 0; i < vec.size(); i++) {
      mat.col(i) = vec.at(i);
    }
  }

};

#ifdef BUILD_WITH_qpOASES
TEST_F(ParallelApproach, ToppraQpOASESExample) {
  const bool printInfo = false;

  //#### create toppra ####
  toppra::PathParametrizationAlgorithmPtr algo =
      std::make_shared<toppra::algorithm::TOPPRA>(this->constraints, this->path);
  algo->setN(this->gridN);
  algo->solver(std::make_shared<toppra::solver::qpOASESWrapper>());
  toppra::ReturnCode rc1 = algo->computePathParametrization(0, 0);
  if (printInfo) std::cout << "rc1 = " << int(rc1) << std::endl;
  ASSERT_EQ(rc1, toppra::ReturnCode::OK);

  // toppra::ReturnCode rc2 = algo->computeFeasibleSets();
  // if (printInfo) std::cout << "rc2 = " << int(rc2) << std::endl;
  // ASSERT_EQ(
  //     rc2,
  //     toppra::ReturnCode::ERR_FAIL_FEASIBLE);  // TODO why not ok? => ASSERT_EQ(rc2,
  //                                              // toppra::ReturnCode::OK);

  toppra::ParametrizationData pd = algo->getParameterizationData();
  if (printInfo)
    std::cout << "pd.gridpoints \n " << pd.gridpoints.transpose() << std::endl;
  if (printInfo)
    std::cout << "pd.parametrization \n " << pd.parametrization.transpose()
              << std::endl;
  if (printInfo)
    std::cout << "pd.controllable_sets \n " << pd.controllable_sets
              << std::endl;
  if (printInfo)
    std::cout << "pd.feasible_sets \n " << pd.feasible_sets.transpose() << std::endl;
  if (printInfo) std::cout << "pd.ret_code = " << int(pd.ret_code) << std::endl;
  ASSERT_EQ(pd.ret_code, toppra::ReturnCode::OK);

  //#### create constant accelaration parametrizer ####
  toppra::Vector gridpoints =
      pd.gridpoints;  // Grid-points used for solving the discretized problem.
  toppra::Vector vsquared =
      pd.parametrization;  // Output parametrization (squared path velocity)
  std::shared_ptr<toppra::parametrizer::ConstAccel> ca =
      std::make_shared<toppra::parametrizer::ConstAccel>(path, gridpoints, vsquared);
  if (printInfo) std::cout << "ca->validate() = " << ca->validate() << std::endl;
  ASSERT_TRUE(ca->validate());

  Eigen::Matrix<toppra::value_type, 1, 2> interval2;
  interval2 = ca->pathInterval();
  if (printInfo) std::cout << "interval2 = " << interval2 << std::endl;
  // EXPECT_DOUBLE_EQ(interval2(0), 0);
  // EXPECT_DOUBLE_EQ(interval2(1), 2);

  const int length2 = 11;
  toppra::Vector times2 =
      toppra::Vector::LinSpaced(length2, interval2(0), interval2(1));
  toppra::Vectors path_pos2;
  path_pos2 = ca->eval(times2, 0);  // TODO this function call fails
  toppra::Vectors path_vel2;
  path_vel2 = ca->eval(times2, 1);  // TODO this function call fails
  toppra::Vectors path_acc2;
  path_acc2 = ca->eval(times2, 2);  // TODO this function call fails
  ASSERT_EQ(path_pos2.size(), length2);
  ASSERT_EQ(path_pos2.at(0).rows(), numJoints);
  ASSERT_EQ(path_vel2.size(), length2);
  ASSERT_EQ(path_vel2.at(0).rows(), numJoints);
  ASSERT_EQ(path_acc2.size(), length2);
  ASSERT_EQ(path_acc2.at(0).rows(), numJoints);
  Eigen::MatrixXd path_pos2_ = Eigen::MatrixXd::Zero(numJoints, length2);
  Eigen::MatrixXd path_vel2_ = Eigen::MatrixXd::Zero(numJoints, length2);
  Eigen::MatrixXd path_acc2_ = Eigen::MatrixXd::Zero(numJoints, length2);
  this->formatVecToMat(path_pos2, path_pos2_);
  this->formatVecToMat(path_vel2, path_vel2_);
  this->formatVecToMat(path_acc2, path_acc2_);
  if (printInfo) std::cout << "path_pos2_\n " << path_pos2_ << std::endl;
  if (printInfo) std::cout << "path_vel2_\n " << path_vel2_ << std::endl;
  if (printInfo) std::cout << "path_acc2_\n " << path_acc2_ << std::endl;
  if (printInfo) std::cout << "times2 \n " << times2.transpose() << std::endl;

  //#### check constraints ####
  for (int jointID = 0; jointID < numJoints; jointID++) {
    for (int i = 0; i < length2; i++) {
      ASSERT_GE(path_vel2_(jointID, i), this->velLimitLower(jointID) * (1 + RTOL));
      ASSERT_LE(path_vel2_(jointID, i), this->velLimitUpper(jointID) * (1 + RTOL));
      ASSERT_GE(path_acc2_(jointID, i), this->accLimitLower(jointID) * (1 + RTOL));
      ASSERT_LE(path_acc2_(jointID, i), this->accLimitUpper(jointID) * (1 + RTOL));
    }
  }
}
#endif

TEST_F(ParallelApproach, ToppraSeidelExample) {
  const bool printInfo = false;

  //#### create toppra ####
  toppra::PathParametrizationAlgorithmPtr algo =
      std::make_shared<toppra::algorithm::TOPPRA>(this->constraints, this->path);
  algo->setN(this->gridN);
  algo->solver(std::make_shared<toppra::solver::Seidel>());
  toppra::ReturnCode rc1 = algo->computePathParametrization(0, 0);
  if (printInfo) std::cout << "rc1 = " << int(rc1) << std::endl;
  ASSERT_EQ(rc1, toppra::ReturnCode::OK);

  // toppra::ReturnCode rc2 = algo->computeFeasibleSets();
  // if (printInfo) std::cout << "rc2 = " << int(rc2) << std::endl;
  // ASSERT_EQ(
  //     rc2,
  //     toppra::ReturnCode::ERR_FAIL_FEASIBLE);  // TODO why not ok? => ASSERT_EQ(rc2,
  //                                              // toppra::ReturnCode::OK);

  toppra::ParametrizationData pd = algo->getParameterizationData();
  if (printInfo)
    std::cout << "pd.gridpoints \n " << pd.gridpoints.transpose() << std::endl;
  if (printInfo)
    std::cout << "pd.parametrization \n " << pd.parametrization.transpose()
              << std::endl;
  if (printInfo)
    std::cout << "pd.controllable_sets \n " << pd.controllable_sets
              << std::endl;
  if (printInfo)
    std::cout << "pd.feasible_sets \n " << pd.feasible_sets.transpose() << std::endl;
  if (printInfo) std::cout << "pd.ret_code = " << int(pd.ret_code) << std::endl;
  ASSERT_EQ(pd.ret_code, toppra::ReturnCode::OK);

  //#### create constant accelaration parametrizer ####
  toppra::Vector gridpoints =
      pd.gridpoints;  // Grid-points used for solving the discretized problem.
  toppra::Vector vsquared =
      pd.parametrization;  // Output parametrization (squared path velocity)
  std::shared_ptr<toppra::parametrizer::ConstAccel> ca =
      std::make_shared<toppra::parametrizer::ConstAccel>(path, gridpoints, vsquared);
  if (printInfo) std::cout << "ca->validate() = " << ca->validate() << std::endl;
  ASSERT_TRUE(ca->validate());

  Eigen::Matrix<toppra::value_type, 1, 2> interval2;
  interval2 = ca->pathInterval();
  if (printInfo) std::cout << "interval2 = " << interval2 << std::endl;
  // EXPECT_DOUBLE_EQ(interval2(0), 0);
  // EXPECT_DOUBLE_EQ(interval2(1), 2);

  const int length2 = 11;
  toppra::Vector times2 =
      toppra::Vector::LinSpaced(length2, interval2(0), interval2(1));
  toppra::Vectors path_pos2;
  path_pos2 = ca->eval(times2, 0);  // TODO this function call fails
  toppra::Vectors path_vel2;
  path_vel2 = ca->eval(times2, 1);  // TODO this function call fails
  toppra::Vectors path_acc2;
  path_acc2 = ca->eval(times2, 2);  // TODO this function call fails
  ASSERT_EQ(path_pos2.size(), length2);
  ASSERT_EQ(path_pos2.at(0).rows(), numJoints);
  ASSERT_EQ(path_vel2.size(), length2);
  ASSERT_EQ(path_vel2.at(0).rows(), numJoints);
  ASSERT_EQ(path_acc2.size(), length2);
  ASSERT_EQ(path_acc2.at(0).rows(), numJoints);
  Eigen::MatrixXd path_pos2_ = Eigen::MatrixXd::Zero(numJoints, length2);
  Eigen::MatrixXd path_vel2_ = Eigen::MatrixXd::Zero(numJoints, length2);
  Eigen::MatrixXd path_acc2_ = Eigen::MatrixXd::Zero(numJoints, length2);
  this->formatVecToMat(path_pos2, path_pos2_);
  this->formatVecToMat(path_vel2, path_vel2_);
  this->formatVecToMat(path_acc2, path_acc2_);
  if (printInfo) std::cout << "path_pos2_\n " << path_pos2_ << std::endl;
  if (printInfo) std::cout << "path_vel2_\n " << path_vel2_ << std::endl;
  if (printInfo) std::cout << "path_acc2_\n " << path_acc2_ << std::endl;
  if (printInfo) std::cout << "times2 \n " << times2.transpose() << std::endl;

  //#### check constraints ####
  for (int jointID = 0; jointID < numJoints; jointID++) {
    for (int i = 0; i < length2; i++) {
      ASSERT_GE(path_vel2_(jointID, i), this->velLimitLower(jointID) * (1 + RTOL));
      ASSERT_LE(path_vel2_(jointID, i), this->velLimitUpper(jointID) * (1 + RTOL));
      ASSERT_GE(path_acc2_(jointID, i), this->accLimitLower(jointID) * (1 + RTOL));
      ASSERT_LE(path_acc2_(jointID, i), this->accLimitUpper(jointID) * (1 + RTOL));
    }
  }
}



TEST_F(ParallelApproach, ToppraParallelExample) {
  const bool printInfo = false;

  //#### create toppra ####
  toppra::PathParametrizationAlgorithmPtr algo =
      std::make_shared<toppra::algorithm::TOPPRA>(this->constraints, this->path);
  algo->setN(this->gridN);
  algo->solver(std::make_shared<toppra::solver::SeidelParallel>());
  toppra::ReturnCode rc1 = algo->computePathParametrizationParallel(0, 0);
  if (printInfo) std::cout << "rc1 = " << int(rc1) << std::endl;
  ASSERT_EQ(rc1, toppra::ReturnCode::OK);

  // toppra::ReturnCode rc2 = algo->computeFeasibleSets();
  // if (printInfo) std::cout << "rc2 = " << int(rc2) << std::endl;
  // ASSERT_EQ(rc2, toppra::ReturnCode::OK);

  toppra::ParametrizationData pd = algo->getParameterizationData();
  if (printInfo)
    std::cout << "pd.gridpoints \n " << pd.gridpoints.transpose() << std::endl;
  if (printInfo)
    std::cout << "pd.parametrization \n " << pd.parametrization.transpose()
              << std::endl;
  if (printInfo)
    std::cout << "pd.controllable_sets \n " << pd.controllable_sets
              << std::endl;
  if (printInfo)
    std::cout << "pd.feasible_sets \n " << pd.feasible_sets.transpose() << std::endl;
  if (printInfo) std::cout << "pd.ret_code = " << int(pd.ret_code) << std::endl;
  ASSERT_EQ(pd.ret_code, toppra::ReturnCode::OK);

  //#### create constant accelaration parametrizer ####
  toppra::Vector gridpoints =
      pd.gridpoints;  // Grid-points used for solving the discretized problem.
  toppra::Vector vsquared =
      pd.parametrization;  // Output parametrization (squared path velocity)
  std::shared_ptr<toppra::parametrizer::ConstAccel> ca =
      std::make_shared<toppra::parametrizer::ConstAccel>(path, gridpoints, vsquared);
  if (printInfo) std::cout << "ca->validate() = " << ca->validate() << std::endl;
  ASSERT_TRUE(ca->validate());

  Eigen::Matrix<toppra::value_type, 1, 2> interval2;
  interval2 = ca->pathInterval();
  if (printInfo) std::cout << "interval2 = " << interval2 << std::endl;
  // EXPECT_DOUBLE_EQ(interval2(0), 0);
  // EXPECT_DOUBLE_EQ(interval2(1), 2);

  const int length2 = 11;
  toppra::Vector times2 =
      toppra::Vector::LinSpaced(length2, interval2(0), interval2(1));
  toppra::Vectors path_pos2;
  path_pos2 = ca->eval(times2, 0);  // TODO this function call fails
  toppra::Vectors path_vel2;
  path_vel2 = ca->eval(times2, 1);  // TODO this function call fails
  toppra::Vectors path_acc2;
  path_acc2 = ca->eval(times2, 2);  // TODO this function call fails
  ASSERT_EQ(path_pos2.size(), length2);
  ASSERT_EQ(path_pos2.at(0).rows(), numJoints);
  ASSERT_EQ(path_vel2.size(), length2);
  ASSERT_EQ(path_vel2.at(0).rows(), numJoints);
  ASSERT_EQ(path_acc2.size(), length2);
  ASSERT_EQ(path_acc2.at(0).rows(), numJoints);
  Eigen::MatrixXd path_pos2_ = Eigen::MatrixXd::Zero(numJoints, length2);
  Eigen::MatrixXd path_vel2_ = Eigen::MatrixXd::Zero(numJoints, length2);
  Eigen::MatrixXd path_acc2_ = Eigen::MatrixXd::Zero(numJoints, length2);
  this->formatVecToMat(path_pos2, path_pos2_);
  this->formatVecToMat(path_vel2, path_vel2_);
  this->formatVecToMat(path_acc2, path_acc2_);
  if (printInfo) std::cout << "path_pos2_\n " << path_pos2_ << std::endl;
  if (printInfo) std::cout << "path_vel2_\n " << path_vel2_ << std::endl;
  if (printInfo) std::cout << "path_acc2_\n " << path_acc2_ << std::endl;
  if (printInfo) std::cout << "times2 \n " << times2.transpose() << std::endl;

  //#### check constraints ####
  for (int jointID = 0; jointID < numJoints; jointID++) {
    for (int i = 0; i < length2; i++) {
      ASSERT_GE(path_vel2_(jointID, i), this->velLimitLower(jointID) * (1 + RTOL));
      ASSERT_LE(path_vel2_(jointID, i), this->velLimitUpper(jointID) * (1 + RTOL));
      ASSERT_GE(path_acc2_(jointID, i), this->accLimitLower(jointID) * (1 + RTOL));
      ASSERT_LE(path_acc2_(jointID, i), this->accLimitUpper(jointID) * (1 + RTOL));
    }
  }
}
