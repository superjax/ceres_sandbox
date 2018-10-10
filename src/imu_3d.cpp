#include <fstream>

#include <ceres/ceres.h>
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <gtest/gtest.h>

#include "factors/range_1d.h"
#include "factors/SE3.h"
#include "factors/imu_3d.h"

#include "geometry/xform.h"
#include "geometry/support.h"
#include "simulator.h"

using namespace ceres;
using namespace Eigen;
using namespace std;
using namespace xform;

TEST(Imu3D, SingleWindow)
{
  Simulator multirotor(false);
  multirotor.load("../lib/multirotor_sim/params/sim_params.yaml");

  Vector6d b, bhat;
  b.block<3,1>(0,0) = multirotor.get_accel_bias();
  b.block<3,1>(3,0) = multirotor.get_gyro_bias();
  bhat.setZero();

  Problem problem;

  Eigen::Matrix<double, 7, 2> xhat, x;
  Eigen::Matrix<double, 3, 2> vhat, v;

  xhat.col(0) = multirotor.get_pose().arr_;
  vhat.col(0) = multirotor.dyn_.get_state().segment<3>(dynamics::VX);
  x.col(0) = multirotor.get_pose().arr_;
  v.col(0) = multirotor.dyn_.get_state().segment<3>(dynamics::VX);

  // Anchor origin node (pose and velocity)
  problem.AddParameterBlock(xhat.data(), 7, new XformAutoDiffParameterization());
  problem.SetParameterBlockConstant(xhat.data());
  problem.AddParameterBlock(vhat.data(), 3);
  problem.SetParameterBlockConstant(vhat.data());

  // Declare the bias parameters
  problem.AddParameterBlock(bhat.data(), 6);

  std::vector<Simulator::measurement_t, Eigen::aligned_allocator<Simulator::measurement_t>> meas_list;
  multirotor.get_measurements(meas_list);

  Imu3DFactorCostFunction *factor = new Imu3DFactorCostFunction(0, bhat, multirotor.get_imu_noise_covariance());

  // Integrate for 1 second
  while (multirotor.t_ < 1.0)
  {
    multirotor.run();
    factor->integrate(multirotor.t_, multirotor.get_imu_prev());
    multirotor.get_measurements(meas_list);
  }

  factor->estimate_xj(xhat.data(), vhat.data(), xhat.data()+7, vhat.data()+3);
  factor->finished();

  // Declare the new parameters
  problem.AddParameterBlock(xhat.data()+7, 7, new XformAutoDiffParameterization());
  problem.AddParameterBlock(vhat.data()+3, 3);

  // Add IMU factor to graph
  problem.AddResidualBlock(new Imu3DFactorAutoDiff(factor), NULL, xhat.data(), xhat.data()+7, vhat.data(), vhat.data()+3, bhat.data());

  // Add measurement of final pose
  x.col(1) = multirotor.get_pose().arr_;
  Vector6d P = (Vector6d() << 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3).finished();
  problem.AddResidualBlock(new XformNodeFactorAutoDiff(new XformNodeFactorCostFunction(x.col(1), P.asDiagonal())), NULL, xhat.data()+7);


  Solver::Options options;
  options.max_num_iterations = 25;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = true;
  Solver::Summary summary;

  cout << "x\n" << x.transpose() << endl;
  cout << "xhat0\n" << xhat.transpose() << endl;
  cout << "b\n" << b.transpose() << endl;
  cout << "bhat\n" << bhat.transpose() << endl;

  ceres::Solve(options, &problem, &summary);
  summary.FullReport();

  cout << "xhat0\n" << xhat.transpose() << endl;
  cout << "bhat\n" << bhat.transpose() << endl;


}
