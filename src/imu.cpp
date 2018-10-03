#include <fstream>

#include <ceres/ceres.h>
#include <Eigen/Dense>
#include <Eigen/Cholesky>

#include "gtest/gtest.h"

#include "factors/range_1d.h"

#include "utils/robot1d.h"
#include "factors/imu.h"

using namespace ceres;
using namespace Eigen;
using namespace std;

TEST(IMU, 1DRobotSLAM)
{
  double ba = 0.1;
  double bahat = 0.0;

  Eigen::Matrix2d Q = (Eigen::Matrix2d() << 0.001, 0, 0, 1e-6).finished();

  Robot1D Robot(ba, Q);
  Robot.waypoints_ = {3, 0, 3, 0};

  const int num_windows = 5;
  int window_size = 500;
  double dt = 0.001;

  Eigen::Matrix<double, 4, 1> landmarks;
  landmarks << 11, 12, 51, 13;
  Eigen::Matrix<double, 4,1> lhat;
  lhat << 5.0, 13.0, -2.0, 12.0;
  lhat = landmarks;

  double rvar = 1.0;
  std::default_random_engine gen;
  std::normal_distribution<double> normal(0.0,1.0);

  Problem problem;

  Eigen::Matrix<double, 3, num_windows> xhat;
  Eigen::Matrix<double, 3, num_windows> x;


  // Initialize the Graph
  xhat(0,0) = Robot.xhat_;
  xhat(1,0) = Robot.vhat_;
  xhat(2,0) = bahat;
  x(0,0) = Robot.x_;
  x(1,0) = Robot.v_;
  x(2,0) = Robot.b_;
  problem.AddParameterBlock(xhat.data(), 3);
  problem.SetParameterBlockConstant(xhat.data());


  for (int win = 1; win < num_windows; win++)
  {
    Imu1DFactor *IMUFactor = new Imu1DFactor(Robot.t_, 0, Q);
    while (Robot.t_ < win*window_size*dt)
    {
      Robot.step(dt);
      IMUFactor->integrate(Robot.t_, Robot.ahat_);
    }

    // Save the actual state
    x(0, win) = Robot.x_;
    x(1, win) = Robot.v_;
    x(2, win) = Robot.b_;

    // Guess at next state
    xhat.col(win) = IMUFactor->estimate_xj(xhat.col(win-1));
    IMUFactor->finished(); // Calculate the Information matrix in preparation for optimization

    // Add landmark measurements
    for (int l = 0; l < landmarks.size(); l++)
    {
      double rbar = (landmarks[l] - Robot.x_) + normal(gen)*sqrt(rvar);
      problem.AddResidualBlock(new Range1dFactorVelocity(rbar, rvar), NULL, lhat.data()+l, xhat.data() + 3*win);
    }

    // Add preintegrated IMU
    problem.AddResidualBlock(IMUFactor, NULL, xhat.data()+3*(win-1), xhat.data()+3*win);
  }


  Solver::Options options;
  options.max_num_iterations = 25;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  options.minimizer_progress_to_stdout = true;

  Solver::Summary summary;

  cout << "ba = " << ba << endl;
  cout << "x: \n" << x <<endl;
  cout << "l: \n" << landmarks.transpose() << endl;
  cout << "bahat = " << bahat << endl;
  cout << "xhat: \n" << xhat << endl;
  cout << "l: \n" << lhat.transpose() << endl;
  ceres::Solve(options, &problem, &summary);
  cout << "bahat = " << bahat << endl;
  cout << "xhat: \n" << xhat << endl;
  cout << "lhat: \n" << lhat.transpose() << endl;

  EXPECT_NEAR(bahat, -ba, 1e-1);

}

