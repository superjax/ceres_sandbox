#include <fstream>

#include <ceres/ceres.h>
#include <Eigen/Dense>
#include <Eigen/Cholesky>

#include "gtest/gtest.h"

#include "factors/range_1d.h"
#include "factors/position_1d.h"
#include "factors/imu_1d.h"
#include "factors/SE3.h"
#include "factors/imu_3d.h"

#include "utils/robot1d.h"
#include "utils/jac.h"

#include "multirotor_sim/simulator.h"


using namespace ceres;
using namespace Eigen;
using namespace std;

TEST(TimeOffset, 1DRobotSLAM)
{
  double ba = 0.2;
  double bahat = 0.00;

  double Td = 0.05;
  double Tdhat = 0.0;

  double Q = 1e-3;

  Robot1D Robot(ba, Q, Td);
  Robot.waypoints_ = {3, 0, 3, 0};

  const int num_windows = 15;
  int window_size = 50;
  double dt = 0.01;

  Eigen::Matrix<double, 8, 1> landmarks = (Eigen::Matrix<double, 8, 1>() << -20, -15, -10, -5, 5, 10, 15, 20).finished();
  Eigen::Matrix<double, 8, 1> lhat = (Eigen::Matrix<double, 8, 1>() << -21, -13, -8, -5, 3, 12, 16, 23).finished();

  double rvar = 1e-2;
  std::default_random_engine gen;
  std::normal_distribution<double> normal(0.0,1.0);

  Problem problem;

  Eigen::Matrix<double, 2, num_windows> xhat;
  Eigen::Matrix<double, 2, num_windows> x;


  // Initialize the Graph
  xhat(0,0) = Robot.xhat_;
  xhat(1,0) = Robot.vhat_;
  x(0,0) = Robot.x_;
  x(1,0) = Robot.v_;

  // Tie the graph to the origin
  problem.AddParameterBlock(xhat.data(), 2);
  problem.SetParameterBlockConstant(xhat.data());

  for (int win = 1; win < num_windows; win++)
  {
    Imu1DFactorCostFunction *IMUFactor = new Imu1DFactorCostFunction(Robot.t_, bahat, Q);
    while (Robot.t_ < win*window_size*dt)
    {
      Robot.step(dt);
      IMUFactor->integrate(Robot.t_, Robot.ahat_);
    }

    // Save the actual state
    x(0, win) = Robot.x_;
    x(1, win) = Robot.v_;

    // Guess at next state
    xhat.col(win) = IMUFactor->estimate_xj(xhat.col(win-1));
    IMUFactor->finished(); // Calculate the Information matrix in preparation for optimization

    // Add preintegrated IMU
    problem.AddParameterBlock(xhat.data()+2*win, 2);
    problem.AddResidualBlock(new Imu1DFactorAutoDiff(IMUFactor), NULL, xhat.data()+2*(win-1), xhat.data()+2*win, &bahat);

    // Add landmark measurements
    for (int l = 0; l < landmarks.size(); l++)
    {
      double rbar = (landmarks[l] - Robot.x_) + normal(gen)*std::sqrt(rvar);
      problem.AddResidualBlock(new Range1dFactorVelocity(rbar, rvar), NULL, lhat.data()+l, xhat.data()+2*win);
    }

    // Add lagged position measurement
    double xbar_lag = Robot.hist_.front().x;
    double xbar_lag_cov = 1e-8;
    problem.AddResidualBlock(new Position1dFactorWithTimeOffset(xbar_lag, xbar_lag_cov), NULL, xhat.data()+2*win, &Tdhat);
  }

  Solver::Options options;
  options.max_num_iterations = 100;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = false;

  Solver::Summary summary;

  //  cout << "xhat0: \n" << xhat << endl;
  //  cout << "lhat0: \n" << lhat.transpose() << endl;
  //  cout << "bahat0 : " << bahat << endl;
  //  cout << "Tdhat0 : " << bahat << endl;
  //  cout << "e0 : " << (x - xhat).array().abs().sum() << endl;
  ceres::Solve(options, &problem, &summary);
  //  cout << "x: \n" << x <<endl;
  //  cout << "xhat: \n" << xhat << endl;
  //  cout << "bahat : " << bahat << endl;
  //  cout << "Tdhat : " << Tdhat << endl;
  //  cout << "ef : " << (x - xhat).array().abs().sum() << endl;
  //  cout << "lhat: \n" << lhat.transpose() << endl;
  //  cout << "l: \n" << landmarks.transpose() << endl;

  EXPECT_NEAR(bahat, ba, 1e-2);
  EXPECT_NEAR(Tdhat, -Td, 1e-2);

}

TEST(TimeOffset, 3DmultirotorPoseGraph)
{
  google::InitGoogleLogging("Imu3D.MultiWindow");
  Simulator multirotor(false);
  multirotor.load("../params/sim_params.yaml");

#if defined (NDEBUG)
  const int N = 1000;
#else
  const int N = 100;
#endif

  Vector6d b, bhat;
  b.block<3,1>(0,0) = multirotor.get_accel_bias();
  b.block<3,1>(3,0) = multirotor.get_gyro_bias();
  bhat.setZero();
  bhat = b;

  double dt, dthat;
  dt = 0.010;
  dthat = 0.0;

  multirotor.truth_transmission_time_ = dt;
  multirotor.truth_update_rate_ = 5;

  Problem problem;

  Eigen::MatrixXd xhat, x;
  Eigen::MatrixXd vhat, v;
  xhat.resize(7, N+1);
  x.resize(7, N+1);
  vhat.resize(3, N+1);
  v.resize(3, N+1);

  xhat.col(0) = multirotor.get_pose().arr_;
  vhat.col(0) = multirotor.dyn_.get_state().segment<3>(dynamics::VX);
  x.col(0) = multirotor.get_pose().arr_;
  v.col(0) = multirotor.dyn_.get_state().segment<3>(dynamics::VX);

  // Anchor origin node (pose and velocity)
  problem.AddParameterBlock(xhat.data(), 7, new XformAutoDiffParameterization());
  problem.SetParameterBlockConstant(xhat.data());
  problem.AddParameterBlock(vhat.data(), 3);
  problem.SetParameterBlockConstant(vhat.data());

  problem.AddParameterBlock(&dthat, 1);
//  problem.SetParameterBlockConstant(&dthat);

  // Declare the bias parameters
  problem.AddParameterBlock(bhat.data(), 6);

  std::vector<Simulator::measurement_t, Eigen::aligned_allocator<Simulator::measurement_t>> meas_list;
  multirotor.get_measurements(meas_list);

  std::vector<Imu3DFactorCostFunction*> factors;
  factors.push_back(new Imu3DFactorCostFunction(0, bhat, multirotor.get_imu_noise_covariance()));

  // Integrate for N frames
  int node = 0;
  Imu3DFactorCostFunction* factor = factors[node];
  std::vector<double> t;
  t.push_back(multirotor.t_);
  Xformd prev2_pose, prev_pose, current_pose;
  double prev2_t, prev_t, current_t;
  Matrix6d pose_cov = Matrix6d::Constant(0);
  while (node < N)
  {
    multirotor.run();
    factor->integrate(multirotor.t_, multirotor.get_imu_prev());
    multirotor.get_measurements(meas_list);
    bool new_node = false;
    for (auto it = meas_list.begin(); it != meas_list.end(); it++)
    {
      switch(it->type)
      {
      case Simulator::POS:
        pose_cov.block<3,3>(0,0) = it->R;
        current_pose.t_ = it->z;
        new_node = true;
        break;
      case Simulator::ATT:
        pose_cov.block<3,3>(3,3) = it->R;
        current_pose.q_ = it->z;
        new_node = true;
        break;
      default:
        break;
      }
      current_t = multirotor.t_;
    }

    if (new_node)
    {
      t.push_back(multirotor.t_);
      node += 1;

      // estimate next node pose and velocity with IMU preintegration
      factor->estimate_xj(xhat.data()+7*(node-1), vhat.data()+3*(node-1),
                          xhat.data()+7*(node), vhat.data()+3*(node));
      // Calculate the Information Matrix of the IMU factor
      factor->finished();

      // Save off True Pose and Velocity for Comparison
      x.col(node) = multirotor.get_pose().arr_;
      v.col(node) = multirotor.dyn_.get_state().segment<3>(dynamics::VX);

      // Declare the new parameters used for this new node
      problem.AddParameterBlock(xhat.data()+7*node, 7, new XformAutoDiffParameterization());
      problem.AddParameterBlock(vhat.data()+3*node, 3);

      // Add IMU factor to graph
      problem.AddResidualBlock(new Imu3DFactorAutoDiff(factor),
        NULL, xhat.data()+7*(node-1), xhat.data()+7*node, vhat.data()+3*(node-1), vhat.data()+3*node, bhat.data());

      // Start a new Factor
      factors.push_back(new Imu3DFactorCostFunction(multirotor.t_, bhat, multirotor.get_imu_noise_covariance()));
      factor = factors[node];

      Matrix6d P = Matrix6d::Identity() * 0.001;
//      if (node > 1 && node < N-1)
//      {
        Vector6d current_pose_dot;
        current_pose_dot.segment<3>(0) = v.col(node);
        current_pose_dot.segment<3>(3) = multirotor.dyn_.get_state().segment<3>(dynamics::WX);
        problem.AddResidualBlock(new XformTimeOffsetAutoDiff(
          new XformTimeOffsetCostFunction(current_pose.elements(), current_pose_dot, pose_cov)), NULL, xhat.data()+7*node, &dthat);
//      }

//      if (node > 2)
//      {
//        Vector6d prev_pose_dot = (current_pose - prev2_pose) / (current_t - prev2_t);
//        problem.AddResidualBlock(new XformTimeOffsetAutoDiff(
//          new XformTimeOffsetCostFunction(prev_pose, prev_pose_dot, P)),
//            NULL, xhat.data()+7*(node-1), &dthat);
//      }
//      prev2_pose = prev_pose;
//      prev_pose = current_pose;
//      prev2_t = prev_t;
//      prev_t = current_t;
    }
  }

  Solver::Options options;
  options.max_num_iterations = 100;
  options.num_threads = 6;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  options.minimizer_progress_to_stdout = false;
  Solver::Summary summary;
  ofstream truth_file("TimeOffset.MultiWindowConstantBias.truth.log", ios::out);
  ofstream est_file("TimeOffset.MultiWindowConstantBias.est.log", ios::out);
  ofstream est0_file("TimeOffset.MultiWindowConstantBias.est0.log", ios::out);

  for (int i = 0; i <= N; i++)
  {
    est0_file.write((char*)&t[i],sizeof(double));
    est0_file.write((char*)(xhat.data()+7*i),sizeof(double)*7);
    est0_file.write((char*)(vhat.data()+3*i),sizeof(double)*3);
  }

//  cout.flush();

  //    cout << "xhat0\n" << xhat.transpose() << endl;
  //    cout << "bhat0\n" << bhat.transpose() << endl;

  ceres::Solve(options, &problem, &summary);
  double error = (b - bhat).norm();

//  cout << summary.FullReport();
  //    cout << "x\n" << x.transpose() << endl;
  //    cout << "xhat0\n" << xhat.transpose() << endl;
//  cout << "b:     " << b.transpose() << endl;
//  cout << "bhat:  " << bhat.transpose() << endl;
//  cout << "err:   " << (b - bhat).transpose() << endl;

//  cout << "dt:    " << dt << endl;
//  cout << "dthat: " << dthat << endl;
  //    cout << "e " << error << endl;
  EXPECT_LE(error, 0.01);
  EXPECT_LE(fabs(dt - dthat), 0.01);

//  Eigen::Matrix<double, 9, N> final_residuals;

  //    cout << "R\n";
  //    for (int node = 1; node <= N; node++)
  //    {
  //        (*factors[node-1])(xhat.data()+7*(node-1), xhat.data()+7*node,
  //                         vhat.data()+3*(node-1), vhat.data()+3*node,
  //                         bhat.data(),
  //                         final_residuals.data()+9*node);
  //        cout << final_residuals.col(node-1).transpose() << "\n";

  //    }
  //    cout << endl;


  for (int i = 0; i <= N; i++)
  {
    truth_file.write((char*)&t[i],sizeof(double));
    truth_file.write((char*)(x.data()+7*i),sizeof(double)*7);
    truth_file.write((char*)(v.data()+3*i),sizeof(double)*3);
    est_file.write((char*)&t[i],sizeof(double));
    est_file.write((char*)(xhat.data()+7*i),sizeof(double)*7);
    est_file.write((char*)(vhat.data()+3*i),sizeof(double)*3);
  }
  truth_file.close();
  est_file.close();
  google::ShutdownGoogleLogging();
}

