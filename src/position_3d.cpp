#include "position_3d.h"
#include <ceres/ceres.h>
#include "gtest/gtest.h"
#include "Eigen/Dense"

using namespace ceres;
using namespace Eigen;

TEST(Position3D, Optimize)
{
  Vector3d x = (Vector3d() << 1.0, 2.0, 3.0).finished();
  int numObs = 10000;
  Vector3d xhat = (Vector3d() << 4.0, 5.0, -2.0).finished();

  Problem problem;

  Vector3d sample;
  Vector3d mean;
  mean.setZero();
  for (int i = 0; i < numObs; i++)
  {
    sample.setRandom();
    sample *= 5e-2;
    sample += x;
    mean += sample;
    problem.AddResidualBlock(new Position3dFactor(sample.data()), NULL, xhat.data());
  }
  mean /= numObs;

  Solver::Options options;
  options.max_num_iterations = 25;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = true;

  Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  EXPECT_NEAR(xhat[0], x[0], 1e-3);
  EXPECT_NEAR(xhat[1], x[1], 1e-3);
  EXPECT_NEAR(xhat[2], x[2], 1e-3);
}


