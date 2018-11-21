#include <fstream>

#include <ceres/ceres.h>
#include <gtest/gtest.h>
#include <Eigen/Dense>


#include "factors/dynamics_1d.h"
#include "factors/dynamics_3d.h"

using namespace Eigen;

TEST (Control, Robot1d_OptimizeTrajectorySingleWindow)
{
  double x0[2] = {0, 0};
  double xf[2] = {1, 0};

  const int N = 500 ;
  double tmax = 1.0;
  double dt = tmax/(double)N;

  MatrixXd x;
  MatrixXd u;
  x.setZero(2, N);
  u.setZero(1, N);

  ceres::Problem problem;

  double v_init = (xf[0] - x0[0])/tmax;

  for (int i = 0; i < N; i++)
  {
    // add parameter blocks
    problem.AddParameterBlock(x.data() + 2*i, 2);
    problem.AddParameterBlock(u.data() + i, 1);
  }

  // pin initial pose (constraint)
  problem.AddResidualBlock(new PositionVelocityConstraint1DFactor(new PositionVelocityConstraint1D(x0[0], x0[1])), NULL, x.data());
  x.col(0) << x0[0], v_init;
  u(0) = 0;

  problem.AddResidualBlock(new InputCost1DFactor(new InputCost1D()), NULL, u.data());
  for (int i = 1; i < N; i++)
  {
    // dynamics constraint
    problem.AddResidualBlock(new DynamicsContraint1DFactor(new DynamicsConstraint1D(dt)), NULL,
                             x.data()+(i-1)*2, x.data()+i*2, u.data()+i-1, u.data()+i);
    // input cost
    problem.AddResidualBlock(new InputCost1DFactor(new InputCost1D()), NULL, u.data()+i);
    x.col(i) << v_init*i*dt, v_init;
  }

  // pin final pose (constraint)
  problem.AddResidualBlock(new PositionVelocityConstraint1DFactor(new PositionVelocityConstraint1D(xf[0], x0[1])), NULL, x.data()+2*(N-1));
  x.col(N-1) << xf[0], v_init;

  ceres::Solver::Options options;
  options.max_num_iterations = 1000;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  options.minimizer_progress_to_stdout = true;
  ceres::Solver::Summary summary;
  std::ofstream traj0_file("1d_trajectory_states_init.bin");
  traj0_file.write((char*)x.data(), sizeof(double)*x.rows()*x.cols());
  traj0_file.close();

  ceres::Solve(options, &problem, &summary);

  std::ofstream traj_file("1d_trajectory_states.bin");
  std::ofstream input_file("1d_trajectory_input.bin");
  traj_file.write((char*)x.data(), sizeof(double)*x.rows()*x.cols());
  input_file.write((char*)u.data(), sizeof(double)*u.rows()*u.cols());
  traj_file.close();
  input_file.close();
}

TEST (Control, Robot1d_OptimizeTrajectoryMultiWindow)
{

  const int W = 2; // Number of windows
  const int K = 500; // collocation Points per window
  double tmax = 1.0;
  double dt = tmax/(double)K;

  Matrix<double, 2, W+1> x_desired;
  x_desired << 0, 1, 0,
               0, 0, 0;

  MatrixXd x;
  MatrixXd u;
  x.setZero(2, K*W+1);
  u.setZero(1, K*W+1);

  ceres::Problem problem;

  for (int i = 0; i <= K*W; i++)
  {
    // add parameter blocks
    problem.AddParameterBlock(x.data() + 2*i, 2);
    problem.AddParameterBlock(u.data() + i, 1);
  }

  for (int w = 0; w < W; w++)
  {

    Vector2d x0 = x_desired.col(w);
    Vector2d xf = x_desired.col(w+1);

    double v_init = (xf[0] - x0[0])/tmax;


    // pin initial pose (constraint)
    problem.AddResidualBlock(new InputCost1DFactor(new InputCost1D()), NULL, u.data());
    for (int i = 0; i < K; i++)
    {
      int id = i + K*w;
      if (i == 0)
      {
        problem.AddResidualBlock(new PositionVelocityConstraint1DFactor(
                                   new PositionVelocityConstraint1D(x0[0], x0[1])), NULL, x.data() + id*2);
      }

      if (id > 0)
      {
        // dynamics constraint
        problem.AddResidualBlock(new DynamicsContraint1DFactor(new DynamicsConstraint1D(dt)), NULL,
                                 x.data()+(id-1)*2, x.data()+id*2, u.data()+id-1, u.data()+id);
      }
      // input cost
      problem.AddResidualBlock(new InputCost1DFactor(new InputCost1D()), NULL, u.data()+id);
      x.col(id) << x0[0] + v_init*i*dt, v_init;
      u(id) = 0;
    }
  }

  // pin final pose (constraint)
  int id = K*W;
  Vector2d xf = x_desired.rightCols(1);
  problem.AddResidualBlock(new DynamicsContraint1DFactor(new DynamicsConstraint1D(dt)), NULL,
                           x.data()+(id-1)*2, x.data()+id*2, u.data()+id-1, u.data()+id);
  problem.AddResidualBlock(new InputCost1DFactor(new InputCost1D()), NULL, u.data()+id);
  problem.AddResidualBlock(new PositionVelocityConstraint1DFactor(
                           new PositionVelocityConstraint1D(xf[0], xf[1])), NULL, x.data() + id*2);
  x.col(id) << xf[0], 1.0;

  ceres::Solver::Options options;
  options.max_num_iterations = 1000;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  options.minimizer_progress_to_stdout = true;
  ceres::Solver::Summary summary;
  std::ofstream traj0_file("1d_trajectory_multi_states_init.bin");
  traj0_file.write((char*)x.data(), sizeof(double)*x.rows()*x.cols());
  traj0_file.close();

  ceres::Solve(options, &problem, &summary);

  std::ofstream traj_file("1d_trajectory_multi_states.bin");
  std::ofstream input_file("1d_trajectory_multi_input.bin");
  traj_file.write((char*)x.data(), sizeof(double)*x.rows()*x.cols());
  input_file.write((char*)u.data(), sizeof(double)*u.rows()*u.cols());
  traj_file.close();
  input_file.close();
}
