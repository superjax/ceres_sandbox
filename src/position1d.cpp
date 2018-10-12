#include <random>

#include <ceres/ceres.h>
#include <gtest/gtest.h>
#include <Eigen/Dense>

#include "factors/position_1d.h"
#include "factors/position_3d.h"
#include "factors/range_1d.h"
#include "factors/transform_1d.h"


using namespace ceres;
using namespace Eigen;

TEST(Position1D, AveragePoints)
{
  double x = 5.0;
  int numObs = 1000;
  double init_x = 3.0;
  double xhat = init_x;

  Problem problem;

  for (int i = 0; i < numObs; i++)
  {
    double sample = x + (rand() % 1000 - 500)/1000.0;
    problem.AddResidualBlock(new Position1dFactor(sample), NULL, &xhat);
  }

  Solver::Options options;
  options.max_num_iterations = 25;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = false;

  Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  EXPECT_NEAR(xhat, x, 1e-2);
}

TEST(Position1D, AveragePointsWithParameterBlock)
{
  double x = 5.0;
  int numObs = 10000;
  double init_x = 3.0;
  double xhat = init_x;

  Problem problem;
  problem.AddParameterBlock(&xhat, 1);

  for (int i = 0; i < numObs; i++)
  {
    double sample = x + (rand() % 1000 - 500)*1e-4;
    problem.AddResidualBlock(new Position1dFactor(sample), NULL, &xhat);
  }

  Solver::Options options;
  options.max_num_iterations = 25;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = false;

  Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  EXPECT_NEAR(xhat, x, 1e-3);
}

TEST(Position1D, SLAM)
{
  double rvar = 1e-5;
  double evar = 1e-1;

  Eigen::Matrix<double, 8, 1> x;
  x << 0, 1, 2, 3, 4, 5, 6, 7;

  Eigen::Matrix<double, 3, 1> l;
  l << 10.0, 15.0, 13.0;

  Eigen::Matrix<double, 3, 1> lhat;
  lhat << 12.0, 9.0, 15.0;

  Eigen::Matrix<double, 8, 1> xhat;
  xhat(0) = 0;

  std::default_random_engine gen;
  std::normal_distribution<double> normal(0.0,1.0);

  Problem problem;

  // Build up the graph
  for (int i = 0; i < x.rows(); i++)
  {
    if (i > 0)
    {
      double That = (x(i) - x(i-1)) + normal(gen)*std::sqrt(evar);
      xhat(i) = xhat(i-1) + That;
      problem.AddResidualBlock(new Transform1d(That, evar), NULL, xhat.data() + i-1, xhat.data() + i);
    }

    for (int j = 0; j < l.rows(); j++)
    {
      double zbar = (l[j] - x[i]) + normal(gen)*std::sqrt(rvar);
      problem.AddResidualBlock(new Range1dFactor(zbar, rvar), NULL, lhat.data()+j, xhat.data() + i);
    }

  }

  Solver::Options options;
  options.max_num_iterations = 25;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = false;

  Solver::Summary summary;
//  std::cout << "x: " << x.transpose() << "  l: " << l.transpose() << "\n";
  double e0 = (x-xhat).norm() + (l-lhat).norm();
//  std::cout << "x0: " << xhat.transpose() << "  l0: " << lhat.transpose() << "  e: " << e0 << "\n";
  ceres::Solve(options, &problem, &summary);
  double ef = (x-xhat).norm() + (l-lhat).norm();
//  std::cout << "xf: " << xhat.transpose() << "  lf: " << lhat.transpose() << "  e: " << ef << std::endl;
  EXPECT_LT(ef, e0);
}
