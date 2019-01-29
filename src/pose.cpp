#include <fstream>

#include <ceres/ceres.h>
#include <Eigen/Dense>
#include <Eigen/Cholesky>

#include "gtest/gtest.h"

#include "geometry/xform.h"
#include "geometry/support.h"
#include "factors/SE3.h"

#define NUM_ITERS 1
#define PRINT_RESULTS 1

using namespace ceres;
using namespace Eigen;
using namespace std;
using namespace xform;

TEST(Pose3D, AveragePoseAutoDiff)
{
  for (int j = 0; j < NUM_ITERS; j++)
  {
    int numObs = 1000;
    int noise_level = 1e-2;
    Xform<double> x = Xform<double>::Random();
    Vector7d xhat = Xform<double>::Identity().elements();

    Problem problem;

    problem.AddParameterBlock(xhat.data(), 7, new XformParamAD());

    for (int i = 0; i < numObs; i++)
    {
      Vector7d sample = (x + Vector6d::Random()*noise_level).elements();
      problem.AddResidualBlock(new XformFactorAD(new XformFunctor(sample.data())), NULL, xhat.data());
    }

    Solver::Options options;
    options.max_num_iterations = 25;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;


    Solver::Summary summary;
    if (j == -1)
    {
      std::cout << "x:   " << x << "\n";
      std::cout << "xhat:" << Xform<double>(xhat) << "\n";
    }
    ceres::Solve(options, &problem, &summary);
    if (j == -1)
      std::cout << "xhat: " << Xform<double>(xhat) << std::endl;
    double error = (Xform<double>(xhat) - x).array().abs().sum();
    EXPECT_LE(error, 1e-8);
  }
}

TEST(Pose3D, GraphSLAM)
{
  // Input between nodes
  Vector6d u;
  u << 1.0, 0.0, 0.0, 0.0, 0.0, 0.1;

  // Odom covariance
  Matrix6d odomcov = Matrix6d::Identity();
  odomcov.block<3,3>(0,0) *= 0.5; // delta position noise
  odomcov.block<3,3>(3,3) *= 0.001; // delta attitude noise
  Matrix6d sqrtodomcov (odomcov.llt().matrixL());

  // LC covariance
  Matrix6d lccov = Matrix6d::Identity();
  lccov.block<3,3>(0,0) *= 0.001; // delta position noise
  lccov.block<3,3>(3,3) *= 0.0001; // delta attitude noise
  Matrix6d sqrtlccov (lccov.llt().matrixL());

  // Simulate the robot while building up the graph
  const int num_steps = 50;
  int num_lc = 12;

  // Raw buffer to hold estimates and truth
  Eigen::Matrix<double, 7, num_steps> x, xhat;

  std::default_random_engine gen;
  std::normal_distribution<double> normal(0.0,1.0);

  // Build up loop closures
  std::vector<std::vector<int>> loop_closures;
  std::uniform_int_distribution<int> uniform_int(0.0,num_steps);
  for (int l = 0; l < num_lc; l++)
  {
    int from = uniform_int(gen);
    int to;
    do
    {
      to = uniform_int(gen);
    } while(to == from);
    std::vector<int> lc {from, to};
    loop_closures.push_back(lc);
  }

  Problem problem;

  // Build up Odometry
  for (int k = 0; k < num_steps; k++)
  {
    if (k == 0)
    {
      // Starting position
      xhat.col(0) = Xform<double>::Identity().elements();
      x.col(0) = Xform<double>::Identity().elements();

      // Start by pinning the initial pose to the origin
      problem.AddParameterBlock(xhat.data(), 7, new XformParamAD()); // Tell ceres that this is a Lie Group
      problem.SetParameterBlockConstant(xhat.data());
    }
    else
    {
      int i = k-1;
      int j = k;

      // Move forward
      Xform<double> xi(x.data()+7*i);
      Xform<double> xihat (xhat.data() + 7*i);
      Map<Vector7d> xj(x.data()+7*j);
      Map<Vector7d> xjhat(xhat.data()+7*j);
      xj = (xi + u).elements();

      // Create measurement (with noise)
      Vector6d noise;
      setNormalRandom(noise, normal, gen);
      noise = sqrtodomcov*noise;
      noise(2) = 0;
      noise(3) = 0;
      noise(4) = 0; // constrain noise to plane
      Vector7d ehat_ij = (Xform<double>::exp(u + noise)).elements();
      xjhat = (xihat * Xform<double>(ehat_ij)).elements();

      // Add odometry edges to graph
      problem.AddParameterBlock(xhat.data() + 7*j, 7, new XformParamAD()); // Tell ceres that this is a lie group
      problem.AddResidualBlock(new XformEdgeFactorAD(new XformEdgeFunctor(ehat_ij, odomcov)), NULL, xhat.data()+7*i, xhat.data()+7*j);
    }
  }

  // Add Loop Closures
  for (int l = 0; l < loop_closures.size(); l++)
  {
    int i = loop_closures[l][0];
    int j = loop_closures[l][1];

    Xform<double> xi(x.data()+7*i);
    Xform<double> xj(x.data()+7*j);
    Xform<double> eij = xi.inverse() * (xj); // true transform

    // Create measurement (with noise)
    Vector6d noise;
    setNormalRandom(noise, normal, gen);
    noise = sqrtlccov*noise;
    noise(2) = 0;
    noise(3) = 0;
    noise(4) = 0; // constrain noise to plane
    Vector7d ehat_ij = eij.elements();

    // Add loop closure edge to graph
    problem.AddResidualBlock(new XformEdgeFactorAD(new XformEdgeFunctor(ehat_ij, lccov)), NULL, xhat.data()+7*i, xhat.data()+7*j);
  }


  Solver::Options options;
  options.max_num_iterations = 25;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  options.minimizer_progress_to_stdout = false;


  Solver::Summary summary;
  ofstream log_file("Pose.GraphSLAM.log", ios::out);

  log_file << "x:\n";
  for (int i = 0; i < num_steps; i++)
  {
    log_file <<"[" << i << "] - " << Xformd(x.col(i)) << "\n";
  }

  log_file << "xhat0:\n";
  for (int i = 0; i < num_steps; i++)
  {
    log_file << "[" << i << "] - " << Xformd(xhat.col(i)) << "\t--\t" << 1/6.0 * (Xformd(xhat.col(i)) - Xformd(x.col(i))).array().square().sum() << "\n";
  }

  // Calculate RMSE
  double RMSE = 0;
  for (int k = 0; k < num_steps; k++)
  {
    RMSE += 1/6.0 * (Xformd(xhat.col(k)) - Xformd(x.col(k))).array().square().sum();
  }
  RMSE /= num_steps;
  log_file << "RMSE0: " << RMSE << endl;

  ceres::Solve(options, &problem, &summary);

  log_file << "xhat:\n:";
  for (int i = 0; i < num_steps; i++)
  {
    log_file << "[" << i << "] - " << Xform<double>(xhat.col(i)) << "\t--\t" << 1/6.0 * (Xformd(xhat.col(i)) - Xformd(x.col(i))).array().square().sum() << "\n";
  }
  // Calculate RMSE
  RMSE = 0;
  for (int k = 0; k < num_steps; k++)
  {
    RMSE += 1/6.0 * (Xformd(xhat.col(k)) - Xformd(x.col(k))).array().square().sum();
  }
  RMSE /= num_steps;
  log_file << "RMSEf: " << RMSE << endl;
  log_file << endl;
  log_file.close();

  EXPECT_LE(RMSE, 0.5);
}
