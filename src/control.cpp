#include <fstream>

#include <ceres/ceres.h>
#include <gtest/gtest.h>
#include <Eigen/Dense>


#include "factors/dynamics_1d.h"
#include "factors/dynamics_3d.h"

using namespace Eigen;
using namespace std;

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
  problem.AddResidualBlock(new PosVel1DFactorAD(new PosVel1DFunctor(x0[0], x0[1])), NULL, x.data());
  x.col(0) << x0[0], v_init;
  u(0) = 0;

  problem.AddResidualBlock(new Input1DFactorAD(new Input1DFunctor()), NULL, u.data());
  for (int i = 1; i < N; i++)
  {
    // dynamics constraint
    problem.AddResidualBlock(new Dyn1dFactorAD(new Dyn1DFunctor(dt)), NULL,
                             x.data()+(i-1)*2, x.data()+i*2, u.data()+i-1, u.data()+i);
    // input cost
    problem.AddResidualBlock(new Input1DFactorAD(new Input1DFunctor()), NULL, u.data()+i);
    x.col(i) << v_init*i*dt, v_init;
  }

  // pin final pose (constraint)
  problem.AddResidualBlock(new PosVel1DFactorAD(new PosVel1DFunctor(xf[0], x0[1])), NULL, x.data()+2*(N-1));
  x.col(N-1) << xf[0], v_init;

  ceres::Solver::Options options;
  options.max_num_iterations = 1000;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  options.minimizer_progress_to_stdout = false;
  ceres::Solver::Summary summary;
  std::ofstream traj0_file("/tmp/ceres_sandbox/1d_trajectory_states_init.bin");
  traj0_file.write((char*)x.data(), sizeof(double)*x.rows()*x.cols());
  traj0_file.close();

  ceres::Solve(options, &problem, &summary);

  std::ofstream traj_file("/tmp/ceres_sandbox/1d_trajectory_states.bin");
  std::ofstream input_file("/tmp/ceres_sandbox/1d_trajectory_input.bin");
  traj_file.write((char*)x.data(), sizeof(double)*x.rows()*x.cols());
  input_file.write((char*)u.data(), sizeof(double)*u.rows()*u.cols());
  traj_file.close();
  input_file.close();
}

TEST (Control, Robot1d_OptimizeTrajectoryMultiWindow)
{

  const int W = 8; // Number of windows
  const int K = 500; // collocation Points per window
  double tmax = 1.0;
  double dt = tmax/(double)K;

  Matrix<double, 2, W+1> x_desired;
  x_desired << 0,  1, 0, 3, -1,  0, -4,  0, 5,
      0, -6, 0, 9,  0, -9,  0, 15, 0;

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
    problem.AddResidualBlock(new Input1DFactorAD(new Input1DFunctor()), NULL, u.data());
    for (int i = 0; i < K; i++)
    {
      int id = i + K*w;
      if (i == 0)
      {
        problem.AddResidualBlock(new PosVel1DFactorAD(
                                   new PosVel1DFunctor(x0[0], x0[1])), NULL, x.data() + id*2);
      }

      if (id > 0)
      {
        // dynamics constraint
        problem.AddResidualBlock(new Dyn1dFactorAD(new Dyn1DFunctor(dt)), NULL,
                                 x.data()+(id-1)*2, x.data()+id*2, u.data()+id-1, u.data()+id);
      }
      // input cost
      problem.AddResidualBlock(new Input1DFactorAD(new Input1DFunctor()), NULL, u.data()+id);
      x.col(id) << x0[0] + v_init*i*dt, v_init;
      u(id) = 0;
    }
  }

  // pin final pose (constraint)
  int id = K*W;
  Vector2d xf = x_desired.rightCols(1);
  problem.AddResidualBlock(new Dyn1dFactorAD(new Dyn1DFunctor(dt)), NULL,
                           x.data()+(id-1)*2, x.data()+id*2, u.data()+id-1, u.data()+id);
  problem.AddResidualBlock(new Input1DFactorAD(new Input1DFunctor()), NULL, u.data()+id);
  problem.AddResidualBlock(new PosVel1DFactorAD(
                             new PosVel1DFunctor(xf[0], xf[1])), NULL, x.data() + id*2);
  x.col(id) << xf[0], 1.0;

  ceres::Solver::Options options;
  options.max_num_iterations = 1000;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  options.minimizer_progress_to_stdout = false;
  ceres::Solver::Summary summary;
  std::ofstream traj0_file("/tmp/ceres_sandbox/1d_trajectory_multi_states_init.bin");
  traj0_file.write((char*)x.data(), sizeof(double)*x.rows()*x.cols());
  traj0_file.close();

  ceres::Solve(options, &problem, &summary);

  std::ofstream traj_file("/tmp/ceres_sandbox/1d_trajectory_multi_states.bin");
  std::ofstream input_file("/tmp/ceres_sandbox/1d_trajectory_multi_input.bin");
  traj_file.write((char*)x.data(), sizeof(double)*x.rows()*x.cols());
  input_file.write((char*)u.data(), sizeof(double)*u.rows()*u.cols());
  traj_file.close();
  input_file.close();
}

TEST (DISABLED_Control, MultirotorSingleStep)
{
  typedef Matrix<double, 10, 1> Vec10;
  Vec10 x0 = (Vec10() << 0, 0, 0, 1, 0, 0, 0, 0, 0, 0).finished();
  Vec10 xf = (Vec10() << 1, 0, 0, 1, 0, 0, 0, 0, 0, 0).finished();

  Matrix4d R;
  R << 1e6, 0, 0, 0,
      0, 1e6, 0, 0,
      0, 0, 1e6, 0,
      0, 0, 0, 1e6;
  double dyn_cost = 1e6;
  MatrixXd x, u;
  x.setZero(10, 2);
  u.setZero(4, 2);
  double dt = 0.1;
  u.row(3).setConstant(0.5);

  ceres::Problem problem;

  problem.AddParameterBlock(x.data(), 10, new Dynamics3DPlusParamAD());
  problem.AddParameterBlock(x.data()+10, 10, new Dynamics3DPlusParamAD());
  problem.AddParameterBlock(u.data(), 4);

  x.col(0) = x0;
  x.col(1) = xf;

  problem.AddResidualBlock(new PosVel3DFactorAD(
      new PosVel3DFunctor(x0.segment<3>(0), 0, dyn_cost)), NULL, x.data());
  problem.AddResidualBlock(new PosVel3DFactorAD(
      new PosVel3DFunctor(xf.segment<3>(0), 0, dyn_cost)), NULL, x.data()+10);
  problem.AddResidualBlock(new Dyn3DFactorAD(new Dyn3dFunctor(dt, dyn_cost)), NULL,
                           x.data(), x.data()+10, u.data(), u.data()+4);
  problem.AddResidualBlock(new Input3DFactorAD(new Input3DFunctor(R)), NULL, u.data());
  problem.AddResidualBlock(new Input3DFactorAD(new Input3DFunctor(R)), NULL, u.data()+4);

  ceres::Solver::Options options;
  options.max_num_iterations = 1000;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  options.minimizer_progress_to_stdout = true;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
}

TEST (DISABLED_Control, Multirotor3d_OptimizeTrajectorySingleWindow)
{
  typedef Matrix<double, 10, 1> Vec10;
  Vec10 x0 = (Vec10() << 0,   0, 0, 1, 0, 0, 0, 0, 0, 0).finished();
  Vec10 xf = (Vec10() << 0.2, 0, 0, 1, 0, 0, 0, 0, 0, 0).finished();

  Matrix4d R;
  R << 1, 0, 0, 0,
       0, 1, 0, 0,
       0, 0, 1, 0,
       0, 0, 0, 2;
  double dyn_cost = 1e6;
  double pos_cost = 1e6;

  const int N = 100;
  double tmax = 1.0;
  double dt = tmax/(double)(N-1);

  MatrixXd x;
  MatrixXd u;
  x.setZero(10, N);
  u.setZero(4, N);
  u.row(3).setConstant(0.5);

  ceres::Problem problem;

//  double v_init = (xf.segment<3>(0) - x0.segment<3>(0)).norm()/tmax;

  for (int i = 0; i < N; i++)
  {
    // add parameter blocks
    problem.AddParameterBlock(x.data() + 10*i, 10, new Dynamics3DPlusParamAD());
    problem.AddParameterBlock(u.data() + 4*i, 4);
    problem.SetParameterUpperBound(u.data() + 4*i, 3, 1.0);
    problem.SetParameterLowerBound(u.data() + 4*i, 3, 0.0);
  }
//  problem.SetParameterBlockConstant(x.data());
//  problem.SetParameterBlockConstant(x.data() + (N-1)*10);

  // pin initial pose (constraint)
  problem.AddResidualBlock(new PosVel3DFactorAD(
                             new PosVel3DFunctor(x0.segment<3>(0), 0, pos_cost)), NULL, x.data());
  x.col(0) = x0;

  problem.AddResidualBlock(new Input3DFactorAD(new Input3DFunctor(R)), NULL, u.data());
  for (int i = 1; i < N; i++)
  {
    // dynamics constraint
    problem.AddResidualBlock(new Dyn3DFactorAD(new Dyn3dFunctor(dt, dyn_cost)), NULL,
                             x.data()+(i-1)*10, x.data()+i*10, u.data()+(i-1)*4, u.data()+i*4);
    // input cost
    problem.AddResidualBlock(new Input3DFactorAD(new Input3DFunctor(R)), NULL, u.data()+4*i);
    x.col(i) = x0;
    x.block<3,1>(0,i) = (xf.segment<3>(0) - x0.segment<3>(0))/tmax * (double)i*dt + x0.segment<3>(0);
//    x(3,i) = 0;
  }

  // pin final pose (constraint)
  Vector3d pf = xf.segment<3>(0);
  problem.AddResidualBlock(new PosVel3DFactorAD(
        new PosVel3DFunctor(pf, 0.0, pos_cost)), NULL, x.data()+10*(N-1));
  x.col(N-1) << xf;

  ceres::Solver::Options options;
  options.max_num_iterations = 1000;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  options.minimizer_progress_to_stdout = true;
  ceres::Solver::Summary summary;
  std::ofstream traj0_file("/tmp/ceres_sandbox/3d_trajectory_states_init.bin");
  traj0_file.write((char*)x.data(), sizeof(double)*x.rows()*x.cols());
  traj0_file.close();
//  cout << "x0 = \n" << x << endl;

  for (int i = 0; i < 3; i++)
  {
    ceres::Solve(options, &problem, &summary);
    dyn_cost *= 10.0;
    pos_cost *= 10.0;
  }
  cout << summary.FullReport() << endl;

//  cout << "xf = \n" << x << endl;
  std::ofstream traj_file("/tmp/ceres_sandbox/3d_trajectory_states.bin");
  std::ofstream input_file("/tmp/ceres_sandbox/3d_trajectory_input.bin");
  traj_file.write((char*)x.data(), sizeof(double)*x.rows()*x.cols());
  input_file.write((char*)u.data(), sizeof(double)*u.rows()*u.cols());
  traj_file.close();
  input_file.close();
}
