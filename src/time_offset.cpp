#include <fstream>

#include <ceres/ceres.h>
#include <Eigen/Dense>
#include <Eigen/Cholesky>

#include "gtest/gtest.h"

#include "factors/range_1d.h"
#include "factors/position_1d.h"

#include "utils/robot1d.h"
#include "factors/imu_1d.h"
#include "utils/jac.h"

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


//TEST
