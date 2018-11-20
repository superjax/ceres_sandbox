#include <fstream>

#include <ceres/ceres.h>
#include <gtest/gtest.h>
#include <Eigen/Dense>


#include "factors/dynamics_1d.h"

using namespace Eigen;

TEST (Control, Robot1D_OptimizeTrajectory)
{
  double x0[2] = {0, 0};
  double xf[2] = {1, 0};

  const int N = 500;
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
