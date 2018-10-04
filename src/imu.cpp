#include <fstream>

#include <ceres/ceres.h>
#include <Eigen/Dense>
#include <Eigen/Cholesky>

#include "gtest/gtest.h"

#include "factors/range_1d.h"

#include "utils/robot1d.h"
#include "factors/imu.h"
#include "utils/jac.h"

using namespace ceres;
using namespace Eigen;
using namespace std;

#define NUM_ITERS 1

TEST(IMU, CheckJacobians)
{
  for (int j = 0; j < NUM_ITERS; j++)
  {
    Vector3d x1, x2;
    double* _x[2];
    _x[0] = x1.data();
    _x[1] = x2.data();

    Eigen::Matrix<double, 3,3, RowMajor> J1, J2;
    Eigen::Matrix<double, 3,3> JFD1, JFD2;
    double* _j[2];
    _j[0] = J1.data();
    _j[1] = J2.data();



    // create the factor and integrate a window of imu data
    double ba = 0.1;
    double bahat = 0.09;
    Eigen::Matrix2d Q = (Eigen::Matrix2d() << 0.001, 0, 0, 1e-6).finished();
    Robot1D Robot(ba, Q);
    Robot.waypoints_ = {3, 0, 3, 0};
    Imu1DFactor* factor = new Imu1DFactor(0, bahat, Q);
    x1(0) = Robot.xhat_;
    x1(1) = Robot.vhat_;
    x1(2) = bahat;
    double dt = 0.001;
    for (int i = 0; i < 500; i++)
    {
      Robot.step(dt);
      factor->integrate(Robot.t_, Robot.ahat_);
    }
    x2 = factor->estimate_xj(x1);
    factor->finished();



    // Get Analytical Jacobians
    Vector3d r;
    factor->Evaluate(_x, r.data(), _j);



    // Perform Finite Differencing
    auto f1 = [factor, x2](MatrixXd x)
    {
      const double* _x[2];
      _x[0] = x.data();
      _x[1] = x2.data();
      Vector3d y;
      factor->Evaluate(_x, y.data(), NULL);
      return y;
    };
    auto f2 = [factor, x1](MatrixXd x)
    {
      const double* _x[2];
      _x[0] = x1.data();
      _x[1] = x.data();
      Vector3d y;
      factor->Evaluate(_x, y.data(), NULL);
      return y;
    };
    JFD1 = calc_jac(f1, x1);
    JFD2 = calc_jac(f2, x2);
    [](){};

    cout << "J1: \n" << J1 << "\n";
    cout << "JFD1: \n" << JFD1 << "\n";
    cout << "J2: \n" << J2 << "\n";
    cout << "JFD2: \n" << JFD2 << "\n";
  }
}

TEST(IMU, 1DRobotSLAMAutoDiff)
{
  double ba = 0.02;
  double bahat = 0.00;

  Eigen::Matrix2d Q = (Eigen::Matrix2d() << 0.01, 0, 0, 1e-6).finished();

  Robot1D Robot(ba, Q);
  Robot.waypoints_ = {3, 0, 3, 0};

  const int num_windows = 15;
  int window_size = 500;
  double dt = 0.001;

  Eigen::Matrix<double, 4, 1> landmarks;
  landmarks << 11, 12, 51, 13;
  Eigen::Matrix<double, 4,1> lhat;
  lhat << 5.0, 13.0, 30.0, 12.0;

  double rvar = 1e-2;
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
  double x0[2] = {Robot.x_, Robot.v_};

  // Tie the graph to the origin
  problem.AddParameterBlock(x0, 2);
  problem.SetParameterBlockConstant(x0);
  problem.AddParameterBlock(xhat.data(), 3);
  problem.AddResidualBlock(new Pose1DConstraint((Matrix2d() << 1e-8, 0, 0, 1e-8).finished()), NULL, x0, xhat.data());

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
    x(2, win) = Robot.b_;

    // Guess at next state
    xhat.col(win) = IMUFactor->estimate_xj(xhat.col(win-1));
    IMUFactor->finished(); // Calculate the Information matrix in preparation for optimization

    // Add preintegrated IMU
    problem.AddParameterBlock(xhat.data()+3*win, 3);
    problem.AddResidualBlock(new Imu1DFactorAutoDiff(IMUFactor), NULL, xhat.data()+3*(win-1), xhat.data()+3*win);

    // Add landmark measurements
    for (int l = 0; l < landmarks.size(); l++)
    {
      double rbar = (landmarks[l] - Robot.x_) + normal(gen)*sqrt(rvar);
      problem.AddResidualBlock(new Range1dFactorVelocity(rbar, rvar), NULL, lhat.data()+l, xhat.data()+3*win);
    }

  }


  Solver::Options options;
  options.max_num_iterations = 100;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = true;

  Solver::Summary summary;

  cout << "xhat0: \n" << xhat << endl;
  cout << "lhat0: \n" << lhat.transpose() << endl;
  cout << "e0 : " << (x - xhat).array().abs().sum() << endl;
  ceres::Solve(options, &problem, &summary);
  cout << "x: \n" << x <<endl;
  cout << "xhat: \n" << xhat << endl;
  cout << "ef : " << (x - xhat).array().abs().sum() << endl;
  cout << "lhat: \n" << lhat.transpose() << endl;
  cout << "l: \n" << landmarks.transpose() << endl;
//  cout << "error: \n" << x - xhat << endl;

  EXPECT_NEAR(bahat, -ba, 1e-1);

}

