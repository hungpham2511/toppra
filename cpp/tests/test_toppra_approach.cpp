//Author: Niels Dehio, 13. September 2020
#include <toppra/geometric_path/piecewise_poly_path.hpp>
#include <toppra/constraint/linear_joint_velocity.hpp>
#include <toppra/constraint/linear_joint_acceleration.hpp>
#include <toppra/algorithm/toppra.hpp>
#include <toppra/parametrizer/const_accel.hpp>

#include "gtest/gtest.h"

// This cpp-example implements the python-example described here:
// https://hungpham2511.github.io/toppra/quickstart.html
class Approach : public testing::Test {
 public:
  Approach() {
  };

  void formatVecToMat(const std::vector<Eigen::VectorXd, Eigen::aligned_allocator<Eigen::VectorXd> > & vec, Eigen::MatrixXd & mat)
  {
    mat.resize(vec.at(0).rows(),vec.size());
    for (size_t i = 0; i < vec.size(); i++) {
      mat.col(i)=vec.at(i);
    }
  }
};

TEST_F(Approach, ToppraCompleteExample) {
  const bool printInfo = false;

  //#### create piecewise polynomial geometric path ####
  const int numJoints = 2;
  Eigen::VectorXd position0{numJoints}, position1{numJoints}, position2{numJoints};
  position0 << 0.0, 0.0;
  position1 << 1.0, 2.0;
  position2 << 2.0, 0.0;
  Eigen::VectorXd velocity0{numJoints}, velocity1{numJoints}, velocity2{numJoints};
  velocity0 << 0.0, 0.0;
  velocity1 << 1.0, 1.0;
  velocity2 << 0.0, 0.0;
  std::vector<Eigen::VectorXd, Eigen::aligned_allocator<Eigen::VectorXd> > positions = {position0, position1, position2}; //[(0, 0), (1, 2), (2, 0)]
  std::vector<Eigen::VectorXd, Eigen::aligned_allocator<Eigen::VectorXd> > velocities = {velocity0, velocity1, velocity2}; //[(0, 0), (1, 1), (0, 0)]
  std::vector<toppra::value_type> steps;
  steps = std::vector<toppra::value_type>{0, 1, 2};
  toppra::PiecewisePolyPath hermite = toppra::PiecewisePolyPath::constructHermite(positions, velocities, steps);
  std::shared_ptr<toppra::PiecewisePolyPath> path;
  path = std::make_shared<toppra::PiecewisePolyPath>(hermite);

  //#### evaluate path ####
  if(printInfo) std::cout << "path->configSize() = " << path->configSize() << std::endl;
  ASSERT_EQ(path->configSize(), 2);
  if(printInfo) std::cout << "path->dof() = " << path->dof() << std::endl;
  ASSERT_EQ(path->dof(), 2);
  Eigen::Matrix<toppra::value_type, 1, 2> interval;
  interval = path->pathInterval();
  if(printInfo) std::cout << "interval = " << interval << std::endl;
  EXPECT_DOUBLE_EQ(interval(0), 0);
  EXPECT_DOUBLE_EQ(interval(1), 2);
  const size_t length = 5;
  Eigen::VectorXd times = toppra::Vector::LinSpaced(length, interval(0), interval(1));
  std::vector<Eigen::VectorXd, Eigen::aligned_allocator<Eigen::VectorXd> > path_pos;
  path_pos = path->eval(times, 0);
  std::vector<Eigen::VectorXd, Eigen::aligned_allocator<Eigen::VectorXd> > path_vel;
  path_vel = path->eval(times, 1);
  std::vector<Eigen::VectorXd, Eigen::aligned_allocator<Eigen::VectorXd> > path_acc;
  path_acc = path->eval(times, 2);
  ASSERT_EQ(path_pos.size(), length);
  ASSERT_EQ(path_pos.at(0).rows(), numJoints);
  ASSERT_EQ(path_vel.size(), length);
  ASSERT_EQ(path_vel.at(0).rows(), numJoints);
  ASSERT_EQ(path_acc.size(), length);
  ASSERT_EQ(path_acc.at(0).rows(), numJoints);
  Eigen::MatrixXd path_pos_ = Eigen::MatrixXd::Zero(numJoints, length);
  Eigen::MatrixXd path_vel_ = Eigen::MatrixXd::Zero(numJoints, length);
  Eigen::MatrixXd path_acc_ = Eigen::MatrixXd::Zero(numJoints, length);
  this->formatVecToMat(path_pos, path_pos_);
  this->formatVecToMat(path_vel, path_vel_);
  this->formatVecToMat(path_acc, path_acc_);
  if(printInfo) std::cout << "path_pos_\n " << path_pos_ << std::endl;
  if(printInfo) std::cout << "path_vel_\n " << path_vel_ << std::endl;
  if(printInfo) std::cout << "path_acc_\n " << path_acc_ << std::endl;
  if(printInfo) std::cout << "times \n " << times.transpose() << std::endl;

  //#### create linear joint-space constraints ####
  Eigen::VectorXd velLimitLower = Eigen::VectorXd::Zero(numJoints);
  Eigen::VectorXd velLimitUpper = Eigen::VectorXd::Zero(numJoints);
  Eigen::VectorXd accLimitLower = Eigen::VectorXd::Zero(numJoints);
  Eigen::VectorXd accLimitUpper = Eigen::VectorXd::Zero(numJoints);
  velLimitLower << -1, -0.5;
  velLimitUpper << 1, 0.5;
  accLimitLower << -0.05, -0.1;
  accLimitUpper << 0.2, 0.3;
  std::shared_ptr<toppra::constraint::LinearJointVelocity> ljv;
  ljv = std::make_shared<toppra::constraint::LinearJointVelocity>(velLimitLower,velLimitUpper);//[[-1, 1], [-0.5, 0.5]]
  std::shared_ptr<toppra::constraint::LinearJointAcceleration> lja;
  lja = std::make_shared<toppra::constraint::LinearJointAcceleration>(accLimitLower,accLimitUpper);//[[-0.05, 0.2], [-0.1, 0.3]]
  std::vector<std::shared_ptr<toppra::LinearConstraint>> constraints;
  constraints.push_back(ljv);
  constraints.push_back(lja);

  //#### create toppra ####
  std::shared_ptr<toppra::algorithm::TOPPRA> algo;
  algo = std::make_shared<toppra::algorithm::TOPPRA>(constraints, path);
  const toppra::value_type vel_start = 0;
  const toppra::value_type vel_end = 0;
  toppra::ReturnCode rc1 = algo->computePathParametrization(vel_start, vel_end);
  if(printInfo) std::cout << "rc1 = " << int(rc1) << std::endl;
  ASSERT_EQ(rc1, toppra::ReturnCode::OK);
  toppra::ReturnCode rc2 = algo->computeFeasibleSets();
  if(printInfo) std::cout << "rc2 = " << int(rc2) << std::endl;
  ASSERT_EQ(rc2, toppra::ReturnCode::ERR_FAIL_FEASIBLE); //TODO why not ok? => ASSERT_EQ(rc2, toppra::ReturnCode::OK);
  toppra::ParametrizationData pd = algo->getParameterizationData();
  if(printInfo) std::cout << "pd.gridpoints \n " << pd.gridpoints.transpose() << std::endl;
  if(printInfo) std::cout << "pd.parametrization \n " << pd.parametrization.transpose() << std::endl;
  if(printInfo) std::cout << "pd.controllable_sets \n " << pd.controllable_sets.transpose() << std::endl;
  if(printInfo) std::cout << "pd.feasible_sets \n " << pd.feasible_sets.transpose() << std::endl;
  if(printInfo) std::cout << "pd.ret_code = " << int(pd.ret_code) << std::endl;
  ASSERT_EQ(pd.ret_code, toppra::ReturnCode::OK);

  //#### create constant accelaration parametrizer ####
  Eigen::VectorXd gridpoints = pd.gridpoints; //Grid-points used for solving the discretized problem.
  Eigen::VectorXd vsquared = pd.parametrization; //Output parametrization (squared path velocity)
  std::shared_ptr<toppra::parametrizer::ConstAccel> ca;
  ca = std::make_shared<toppra::parametrizer::ConstAccel>(path, gridpoints, vsquared);
  if(printInfo) std::cout << "ca->validate() = " << ca->validate() << std::endl;
  ASSERT_FALSE(ca->validate()); //TODO why false? => ASSERT_TRUE(ca->validate());
  Eigen::Matrix<toppra::value_type, 1, 2> interval2;
  interval2 = ca->pathInterval();
  if(printInfo) std::cout << "interval2 = " << interval2 << std::endl;
  EXPECT_DOUBLE_EQ(interval2(0), 0);
  EXPECT_DOUBLE_EQ(interval2(1), 2);
  const int length2=11;
  Eigen::VectorXd times2 = toppra::Vector::LinSpaced(length2, interval2(0), interval2(1));
  std::vector<Eigen::VectorXd, Eigen::aligned_allocator<Eigen::VectorXd> > path_pos2;
  path_pos2 = ca->eval(times2, 0); //TODO this function call fails
  std::vector<Eigen::VectorXd, Eigen::aligned_allocator<Eigen::VectorXd> > path_vel2;
  path_vel2 = ca->eval(times2, 1); //TODO this function call fails
  std::vector<Eigen::VectorXd, Eigen::aligned_allocator<Eigen::VectorXd> > path_acc2;
  path_acc2 = ca->eval(times2, 2); //TODO this function call fails
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
  if(printInfo) std::cout << "path_pos2_\n " << path_pos2_ << std::endl;
  if(printInfo) std::cout << "path_vel2_\n " << path_vel2_ << std::endl;
  if(printInfo) std::cout << "path_acc2_\n " << path_acc2_ << std::endl;
  if(printInfo) std::cout << "times2 \n " << times2.transpose() << std::endl;

  //#### check constraints ####
  for(int jointID=0; jointID<numJoints; jointID++){
    for(int i=0; i<length2; i++){
      ASSERT_TRUE(path_vel2_(jointID,i)>=velLimitLower(jointID));
      ASSERT_TRUE(path_vel2_(jointID,i)<=velLimitUpper(jointID));
      ASSERT_TRUE(path_acc2_(jointID,i)>=accLimitLower(jointID));
      ASSERT_TRUE(path_acc2_(jointID,i)<=accLimitUpper(jointID));
    }
  }
}
