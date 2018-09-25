#include "lie/quat.h"
#include <ceres/ceres.h>
#include "factors/attitude_3d.h"

#include "gtest/gtest.h"
#include "Eigen/Dense"

using namespace ceres;
using namespace Eigen;
using namespace std;

Eigen::MatrixXd calc_jac(std::function<Eigen::MatrixXd(Eigen::MatrixXd)> fun, Eigen::MatrixXd x)
{
  Eigen::MatrixXd y = fun(x);
  int cols = x.rows();
  int rows = y.rows();

  Eigen::MatrixXd I;
  I.resize(cols, cols);
  I.setZero(cols, cols);
  for (int i = 0; i < cols; i++)
    I(i,i) = 1e-8;

  Eigen::MatrixXd JFD;
  JFD.setZero(rows, cols);
  for (int i =0; i < cols; i++)
  {
    Eigen::MatrixXd xp = x + I.col(i);
    Eigen::MatrixXd xm = x - I.col(i);
    Eigen::MatrixXd yp = fun(xp);
    Eigen::MatrixXd ym = fun(xm);
    JFD.col(i) = (yp - ym)/(2*1e-8);
  }
  return JFD;
}

Eigen::MatrixXd invotimes(Eigen::MatrixXd q2, Eigen::MatrixXd q1)
{
  quat::Quat Q2(q2);
  quat::Quat Q1(q1);
  return Q2.inverse().otimes(Q1).arr_;
}

Eigen::MatrixXd logp1(Eigen::MatrixXd _w, Eigen::MatrixXd xyz)
{
  double w = _w(0,0);
  double nxyz = xyz.norm();
  return  2.0 * atan2(nxyz, w) * xyz / nxyz;
}

Eigen::MatrixXd logp2(Eigen::MatrixXd xyz, Eigen::MatrixXd _w)
{
  double w = _w(0,0);
  double nxyz = xyz.norm();
  return  2.0 * atan2(nxyz, w) * xyz / nxyz;
}

MatrixXd boxminus(MatrixXd q2, MatrixXd q1)
{
  MatrixXd qtilde = invotimes(q2, q1);
  double w = qtilde(0,0);
  Vector3d xyz = qtilde.block<3,1>(1,0);
  double nxyz = xyz.norm();
  return 2.0 * atan2(nxyz, w) * xyz / nxyz;
}

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

TEST(Attitude3d, logp1)
{
  Vector4d q1;
  q1.setRandom(4,1);
  q1 /= q1.norm();
  double w = q1(0,0);
  MatrixXd _w; _w.resize(1,1); _w << w;
  MatrixXd xyz = q1.segment<3>(1);
  double nxyz = xyz.norm();
  MatrixXd JA = -(2.0 * nxyz) / (nxyz*nxyz + w*w) * xyz / nxyz;
  MatrixXd JFD = calc_jac([xyz](MatrixXd w){return logp1(w, xyz);}, _w);
//  cout << "JA = \n" << JA << endl;
//  cout << "JFD = \n" << JFD << endl;
  EXPECT_LE((JA - JFD).array().abs().sum(), 1e-7);
}

TEST(Attitude3d, logp2)
{
  Vector4d q1;
  q1.setRandom(4,1);
  q1 /= q1.norm();
  double w = q1(0,0);
  MatrixXd _w; _w.resize(1,1); _w << w;
  MatrixXd xyz = q1.segment<3>(1);
  double nxyz = xyz.norm();
  Matrix3d I = Eigen::Matrix3d::Identity();
  MatrixXd JA = (2.0 * w) / (nxyz*nxyz + w*w) * (xyz * xyz.transpose()) / (nxyz*nxyz)
      + 2.0 * atan2(nxyz, w) * (I * nxyz*nxyz - xyz * xyz.transpose())/(nxyz*nxyz*nxyz);
  MatrixXd JFD = calc_jac([_w](MatrixXd _xyz){return logp2(_xyz, _w);}, xyz);
//  cout << "JA = \n" << JA << endl;
//  cout << "JFD = \n" << JFD << endl;
  EXPECT_LE((JA - JFD).array().abs().sum(), 1e-7);
}


TEST(Attitude3d, invotimesjac)
{
  Eigen::MatrixXd q1, q2;
  q1.setRandom(4,1);
  q2.setRandom(4,1);
  q1 /= q1.norm();
  q2 /= q2.norm();

  double q1w = q1(0, 0);
  double q1x = q1(1, 0);
  double q1y = q1(2, 0);
  double q1z = q1(3, 0);
  Eigen::MatrixXd J;
  J.resize(4,4);
  J << q1w,  q1x,  q1y,  q1z,
       q1x, -q1w, -q1z,  q1y,
       q1y,  q1z, -q1w, -q1x,
       q1z, -q1y,  q1x, -q1w;
  Eigen::MatrixXd JFD = calc_jac([q1](Eigen::MatrixXd q2){return invotimes(q2, q1);}, q2);
//  cout << "JFD: \n" << JFD << endl;
//  cout << "JA: \n" << J<< endl;
  EXPECT_LE((J - JFD).array().abs().sum(), 1e-7);
}

TEST(Attitude3d, boxminusjac)
{
  Eigen::MatrixXd q1, q2;
  q1.setRandom(4,1);
  q2.setRandom(4,1);
  q1 /= q1.norm();
  q2 /= q2.norm();

  double q1w = q1(0, 0);
  double q1x = q1(1, 0);
  double q1y = q1(2, 0);
  double q1z = q1(3, 0);
  Eigen::MatrixXd Qmat;
  Qmat.resize(4,4);
  Qmat << q1w,  q1x,  q1y,  q1z,
          q1x, -q1w, -q1z,  q1y,
          q1y,  q1z, -q1w, -q1x,
          q1z, -q1y,  q1x, -q1w;
  Vector4d qtilde = invotimes(q2, q1);
  double w = qtilde(0,0);
  Vector3d xyz = qtilde.block<3,1>(1,0);
  double nxyz = xyz.norm();
  Matrix3d I = Eigen::Matrix3d::Identity();

  MatrixXd JA;
  JA.resize(3, 4);
  JA.block<3,1>(0,0) = -(2.0 * nxyz) / (nxyz*nxyz + w*w) * xyz / nxyz;
  JA.block<3,3>(0,1) = (2.0 * w) / (nxyz*nxyz + w*w) * (xyz * xyz.transpose()) / (nxyz*nxyz)
      + 2.0 * atan2(nxyz, w) * (I * nxyz*nxyz - xyz * xyz.transpose())/(nxyz*nxyz*nxyz);
  JA = JA * Qmat;

  MatrixXd JFD = calc_jac([q1](MatrixXd _q2){return boxminus(_q2, q1);}, q2);
//  cout << "JA = \n" << JA << endl;
//  cout << "JFD = \n" << JFD << endl;
  EXPECT_LE((JA - JFD).array().abs().sum(), 1e-7);
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

//  std::cout << "J\n" << J << "\n";
//  std::cout << "JFD\n" << JFD << "\n";
  EXPECT_LE((J - JFD).array().abs().sum(), 1e-6);
}

TEST(Attitude3d, CheckLocalParamJac)
{
  QuatParameterization* param = new QuatParameterization();

  quat::Quat x = quat::Quat::Random();
  Eigen::Vector3d delta;
  delta.setZero(); // Jacobian is evaluated at dx = 0
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

//  std::cout << "J\n" << J << "\n";
//  std::cout << "JFD\n" << JFD << "\n";
  EXPECT_LE((J - JFD).array().abs().sum(), 1e-5);

}

TEST(Attitude3d, AverageAttitude)
{
  int numObs = 1000;
  int noise_level = 1e-2;
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
  EXPECT_LE((xhat - x).array().abs().sum(), 1e-8);
}
