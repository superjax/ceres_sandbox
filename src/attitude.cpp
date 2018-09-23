#include "lie/quat.h"
#include <ceres/ceres.h>
#include "factors/attitude_3d.h"

#include "gtest/gtest.h"
#include "Eigen/Dense"

using namespace ceres;
using namespace Eigen;

TEST(Attitude3d, CheckLocalParamPlus)
{
  for (int i = 0; i < 1000; i++)
  {
    QuatParameterization* param = new QuatParameterization();

    quat::Quat x = quat::Quat::Random();
    Eigen::Vector3d delta;
    delta.setRandom();
    quat::Quat xplus1, xplus2;
    param->Plus(x.data(), delta.data(), xplus1.data());
    xplus2 = x + delta;
    EXPECT_FLOAT_EQ(xplus1.w(), xplus2.w());
    EXPECT_FLOAT_EQ(xplus1.x(), xplus2.x());
    EXPECT_FLOAT_EQ(xplus1.y(), xplus2.y());
    EXPECT_FLOAT_EQ(xplus1.z(), xplus2.z());
  }
}

TEST(Attitude3d, CheckFactorEvaluate)
{
  for (int i = 0; i < 1000; i++)
  {
    quat::Quat x1 = quat::Quat::Random();
    quat::Quat x2 = quat::Quat::Random();
    Vector3d delta1, delta2;
    QuatFactor* factor = new QuatFactor(x1.data());
    double* p[1];
    p[0] = x2.data();
    factor->Evaluate(p, delta1.data(), 0);
    delta2 = x1 - x2;
    EXPECT_FLOAT_EQ(delta1(0), delta2(0));
    EXPECT_FLOAT_EQ(delta1(1), delta2(1));
    EXPECT_FLOAT_EQ(delta1(2), delta2(2));
  }
}

TEST(Attitude3d, CheckFactorJac)
{
  quat::Quat x1 = quat::Quat::Random();
  quat::Quat x2 = quat::Quat::Random();
  Vector3d delta1, delta2;
  QuatFactor* factor = new QuatFactor(x1.data());
  double* p[1];
  p[0] = x2.data();

  Eigen::Matrix<double, 3,4, RowMajor> J, JFD;
  double* j[1];
  j[0] = J.data();


  factor->Evaluate(p, delta1.data(), j);
  delta2 = x1 - x2;

  Eigen::Matrix<double, 4, 4> I4x4 = Eigen::Matrix<double, 4, 4>::Identity();
  I4x4 *= 1e-8;
  Vector4d x2primeplus, x2primeminus;
  Vector3d delta1primeplus, delta1primeminus;
  for (int i = 0; i < 4; i++)
  {
    x2primeplus = x2.arr_ + I4x4.col(i);
    double * pprimeplus[1];
    pprimeplus[0] = x2primeplus.data();
    factor->Evaluate(pprimeplus, delta1primeplus.data(), 0);

    x2primeminus = x2.arr_ - I4x4.col(i);
    double * pprimeminus[1];
    pprimeminus[0] = x2primeminus.data();
    factor->Evaluate(pprimeminus, delta1primeminus.data(), 0);


    JFD.col(i) = (delta1primeplus - delta1primeminus) / 2e-8;
  }

  std::cout << "J\n" << J << "\n";
  std::cout << "JFD\n" << JFD << "\n";
}

TEST(Attitude3d, CheckLocalParamJac)
{
  QuatParameterization* param = new QuatParameterization();

  quat::Quat x = quat::Quat::Random();
  Eigen::Vector3d delta;
  delta.setRandom();
  quat::Quat xplus1, xplus2;
  param->Plus(x.data(), delta.data(), xplus1.data());
  xplus2 = x + delta;

  Eigen::Matrix<double, 4, 3, RowMajor> J, JFD;
  param->ComputeJacobian(x.data(), J.data());

  Matrix3d I3x3 = Matrix3d::Identity() * 1e-8;
  Vector3d deltaprimeplus;
  Vector3d deltaprimeminus;
  Vector4d xplusprimeplus;
  Vector4d xplusprimeminus;
  for (int i = 0; i < 3; i++)
  {
    deltaprimeplus = delta + I3x3.col(i);
    deltaprimeminus = delta - I3x3.col(i);
    param->Plus(x.data(), deltaprimeplus.data(), xplusprimeplus.data());
    param->Plus(x.data(), deltaprimeminus.data(), xplusprimeminus.data());
    JFD.col(i) = (xplusprimeplus - xplusprimeminus)/2e-8;
  }

  EXPECT_LE((J - JFD).array().abs().sum(), 0.5);
}

TEST(Attitude3d, AverageAttitude)
{
  int numObs = 1000;
  int noise_level = 1e-5;
  quat::Quat x = quat::Quat::Random();
  quat::Quat xhat = quat::Quat::Identity();

  Problem problem;
  problem.AddParameterBlock(xhat.data(), 4, new QuatParameterization());

  for (int i = 0; i < numObs; i++)
  {
    quat::Quat sample = x + Vector3d::Random()*noise_level;
    problem.AddResidualBlock(new QuatFactor(sample.data()), NULL, xhat.data());
  }

  Solver::Options options;
  options.max_num_iterations = 25;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = true;

  Solver::Summary summary;
  std::cout << "x:   " << x << "\n";
  std::cout << "xhat:" << xhat << "\n";
  ceres::Solve(options, &problem, &summary);
  std::cout << "xhat: " << xhat << std::endl;
}
