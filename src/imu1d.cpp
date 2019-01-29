#include <fstream>

#include <ceres/ceres.h>
#include <Eigen/Dense>
#include <Eigen/Cholesky>

#include "gtest/gtest.h"

#include "factors/range_1d.h"

#include "utils/robot1d.h"
#include "factors/imu_1d.h"
#include "utils/jac.h"

using namespace ceres;
using namespace Eigen;
using namespace std;

#define NUM_ITERS 1

TEST(Imu1D, 1DRobotSingleWindow)
{
  double ba = 10.0;
  double bahat = 0.00;

  double Q = 1e-6;

  Robot1D Robot(ba, Q);
  Robot.waypoints_ = {3, 0, 3, 0};
  int window_size = 50;
  double dt = 0.01;

  Problem problem;

  Eigen::Matrix<double, 2, 2> xhat;
  Eigen::Matrix<double, 2, 2> x;


  // Initialize the Graph
  xhat(0,0) = Robot.xhat_;
  xhat(1,0) = Robot.vhat_;
  x(0,0) = Robot.x_;
  x(1,0) = Robot.v_;

  // Tie the graph to the origin
  problem.AddParameterBlock(xhat.data(), 2);
  problem.SetParameterBlockConstant(xhat.data());

  Imu1DFunctor *IMUFactor = new Imu1DFunctor(Robot.t_, bahat, Q);
  while (Robot.t_ < window_size*dt)
  {
    Robot.step(dt);
    IMUFactor->integrate(Robot.t_, Robot.ahat_);
  }

  // Save the actual state
  x(0,1) = Robot.x_;
  x(1,1) = Robot.v_;

  // Guess at next state
  xhat.col(1) = IMUFactor->estimate_xj(xhat.col(0));
  IMUFactor->finished(); // Calculate the Information matrix in preparation for optimization

  // Add preintegrated IMU
  problem.AddParameterBlock(xhat.data()+2, 2);
  problem.AddResidualBlock(new Imu1DFactorAD(IMUFactor), NULL, xhat.data(), xhat.data()+2, &bahat);

  // Add measurement of final pose
  Vector2d zj = x.col(1);
  Matrix2d pose_cov = (Matrix2d() << 1e-8, 0, 0, 1e-8).finished();
  problem.AddResidualBlock(new Pose1DFactor(zj, pose_cov), NULL, xhat.data()+2);


  Solver::Options options;
  options.max_num_iterations = 100;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = false;

  Solver::Summary summary;

//  cout << "xhat0: \n" << xhat << endl;
//  cout << "bahat0 : " << bahat << endl;
//  cout << "e0 : " << (x - xhat).array().abs().sum() << endl;
  ceres::Solve(options, &problem, &summary);
//  cout << "x: \n" << x <<endl;
//  cout << "b: \n" << ba <<endl;
//  cout << "xhatf: \n" << xhat << endl;
//  cout << "bhatf: " << bahat << endl;
//  cout << "ef : " << (x - xhat).array().abs().sum() << endl;
  //  cout << "error: \n" << x - xhat << endl;

  EXPECT_NEAR(bahat, ba, 1e-1);
}

TEST(Imu1D, 1DRobotLocalization)
{
  double ba = 10.0;
  double bahat = 0.00;

  double Q = 1e-6;

  Robot1D Robot(ba, Q);
  Robot.waypoints_ = {3, 0, 3, 0};
  const int window_size = 50;
  const int num_windows = 100;
  double dt = 0.01;

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

  for (int i = 1; i < num_windows; i++)
  {
    Imu1DFunctor *IMUFactor = new Imu1DFunctor(Robot.t_, bahat, Q);
    while (Robot.t_ < i*window_size*dt)
    {
      Robot.step(dt);
      IMUFactor->integrate(Robot.t_, Robot.ahat_);
    }

    // Save the actual state
    x(0,i) = Robot.x_;
    x(1,i) = Robot.v_;

    // Guess at next state
    xhat.col(i) = IMUFactor->estimate_xj(xhat.col(i-1));
    IMUFactor->finished(); // Calculate the Information matrix in preparation for optimization

    // Add preintegrated IMU
    problem.AddParameterBlock(xhat.data()+2*i, 2);
    problem.AddResidualBlock(new Imu1DFactorAD(IMUFactor), NULL, xhat.data()+2*(i-1), xhat.data()+2*i, &bahat);
  }

  // Add measurement of final pose
  Vector2d zj = x.col(num_windows-1);
  Matrix2d pose_cov = (Matrix2d() << 1e-8, 0, 0, 1e-8).finished();
  problem.AddResidualBlock(new Pose1DFactor(zj, pose_cov), NULL, xhat.data()+2*(num_windows-1));


  Solver::Options options;
  options.max_num_iterations = 100;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = false;

  Solver::Summary summary;

//  cout << "xhat0: \n" << xhat << endl;
//  cout << "bahat0 : " << bahat << endl;
//  cout << "e0 : " << (x - xhat).array().square().sum()/ << endl;
  ceres::Solve(options, &problem, &summary);
//  cout << "x: \n" << x <<endl;
//  cout << "b: \n" << ba <<endl;
//  cout << "xhatf: \n" << xhat << endl;
//  cout << "bhatf: " << bahat << endl;
//  cout << "ef : " << (x - xhat).array().abs().sum() << endl;
  //  cout << "error: \n" << x - xhat << endl;

  EXPECT_NEAR(bahat, ba, 1e-1);
}

TEST(Imu1D, 1DRobotSLAM)
{
  double ba = 0.2;
  double bahat = 0.00;

  double Q = 1e-3;

  Robot1D Robot(ba, Q);
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
    Imu1DFunctor *IMUFactor = new Imu1DFunctor(Robot.t_, bahat, Q);
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
    problem.AddResidualBlock(new Imu1DFactorAD(IMUFactor), NULL, xhat.data()+2*(win-1), xhat.data()+2*win, &bahat);

    // Add landmark measurements
    for (int l = 0; l < landmarks.size(); l++)
    {
      double rbar = (landmarks[l] - Robot.x_) + normal(gen)*std::sqrt(rvar);
      problem.AddResidualBlock(new RangeVel1DFactor(rbar, rvar), NULL, lhat.data()+l, xhat.data()+2*win);
    }
  }

  Solver::Options options;
  options.max_num_iterations = 100;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = false;

  Solver::Summary summary;

//  cout << "xhat0: \n" << xhat << endl;
//  cout << "lhat0: \n" << lhat.transpose() << endl;
//  cout << "bahat0 : " << bahat << endl;
//  cout << "e0 : " << (x - xhat).array().abs().sum() << endl;
  ceres::Solve(options, &problem, &summary);
//  cout << "x: \n" << x <<endl;
//  cout << "xhat: \n" << xhat << endl;
//  cout << "bahat : " << bahat << endl;
//  cout << "ef : " << (x - xhat).array().abs().sum() << endl;
//  cout << "lhat: \n" << lhat.transpose() << endl;
//  cout << "l: \n" << landmarks.transpose() << endl;

  EXPECT_NEAR(bahat, ba, 1e-2);

}

TEST (Imu1D, dydb)
{
  std::vector<double> a;
  std::vector<double> dt;
  for (int i = 0; i < 1000; i++)
  {
    a.push_back((rand() % 1000)/1000.0 - 0.5);
    dt.push_back(0.01 + ((rand() % 1000)/500.0 - 1.0)/1000.0);
  }
  Vector2d y0 {0.0, 0.0};
  typedef Eigen::Matrix<double, 1, 1> Matrix1d;
  Matrix1d b {(rand() % 1000)/10000.0};

  auto fun = [a, y0, dt](Matrix1d b)
  {
    Matrix2d F;
    Vector2d A, B;
    Vector2d y = y0;
    for (int i = 0; i < a.size(); i++)
    {
      F << 1, dt[i], 0, 1;
      A << 0.5*dt[i]*dt[i], dt[i];
      B << -0.5*dt[i]*dt[i], -dt[i];
      y = F*y + A*a[i] + B*b;
    }
    return y;
  };

  Vector2d J {0, 0};
  Matrix2d F;
  Vector2d B;
  for (int i = 0; i < a.size(); i++)
  {
    F << 1, dt[i], 0, 1;
    B << -0.5*dt[i]*dt[i], -dt[i];
    J = F * J + B;
  }
  Vector2d JFD = calc_jac(fun, b);
  EXPECT_LE((J-JFD).array().abs().sum(), 1e-6);
}
