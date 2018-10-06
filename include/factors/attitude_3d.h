#pragma once

#include <ceres/ceres.h>
#include "geometry/quat.h"

using namespace Eigen;
using namespace quat;

struct QuatPlus {
  template<typename T>
  bool operator()(const T* x, const T* delta, T* x_plus_delta) const
  {
    quat::Quat<T> q(x);
    Map<const Matrix<T,3,1>> d(delta);
    Map<Matrix<T,4,1>> qp(x_plus_delta);

    qp = (q + d).elements();
    return true;
  }
};
typedef ceres::AutoDiffLocalParameterization<QuatPlus, 4, 3> QuatAutoDiffParameterization;

class QuatParameterization : public ceres::LocalParameterization
{
public:
  ~QuatParameterization() {}
  bool Plus(const double* x,
            const double* delta,
            double* x_plus_delta) const
  {
    Quat<double> q(x);
    Map<const Vector3d> d(delta);
    Map<Vector4d> qp(x_plus_delta);
    qp = (q + d).elements();
    return true;
  }

  bool ComputeJacobian(const double* x, double* jacobian) const
  {
    jacobian[0] = -x[1]/2.0; jacobian[1]  = -x[2]/2.0; jacobian[2]  = -x[3]/2.0;
    jacobian[3] =  x[0]/2.0; jacobian[4]  = -x[3]/2.0; jacobian[5]  =  x[2]/2.0;
    jacobian[6] =  x[3]/2.0; jacobian[7]  =  x[0]/2.0; jacobian[8]  = -x[1]/2.0;
    jacobian[9] = -x[2]/2.0; jacobian[10] =  x[1]/2.0; jacobian[11] =  x[0]/2.0;
  }
  int GlobalSize() const {return 4;}
  int LocalSize() const {return 3;}
};

class QuatFactor : public ceres::SizedCostFunction<3,4>
{
public:
  QuatFactor(double *x)
  {
    quat_.arr_ = Map<Vector4d>(x);
  }
  virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const
  {
    Eigen::Map<Eigen::Vector3d> res(residuals);
    quat::Quat<double> q2(parameters[0]);
    quat::Quat<double> qtilde = q2.inverse().otimes(quat_);
    double negative = 1.0;
    if (qtilde.w() < 0.0)
    {
      negative = -1.0;
      qtilde.arr_ *= -1.0;
    }
    res = quat::Quat<double>::log(qtilde);


    if (jacobians)
    {
      if (jacobians[0])
      {
        double w = qtilde.w();
        Map<Vector3d> xyz(qtilde.data() + 1);
        double nxyz = xyz.norm();
        Eigen::Map<Eigen::Matrix<double, 3, 4, RowMajor>> J(jacobians[0]);
        static const Matrix3d I = Eigen::Matrix3d::Identity();

        double q1w = quat_.arr_(0);
        double q1x = quat_.arr_(1);
        double q1y = quat_.arr_(2);
        double q1z = quat_.arr_(3);
        Eigen::Matrix4d Qmat;
        Qmat.resize(4,4);
        Qmat << q1w,  q1x,  q1y,  q1z,
                q1x, -q1w, -q1z,  q1y,
                q1y,  q1z, -q1w, -q1x,
                q1z, -q1y,  q1x, -q1w;
        J.block<3,1>(0,0) = -(2.0 * nxyz) / (nxyz*nxyz + w*w) * xyz / nxyz;
        J.block<3,3>(0,1) = (2.0 * w) / (nxyz*nxyz + w*w) * (xyz * xyz.transpose()) / (nxyz*nxyz)
            + 2.0 * atan2(nxyz, w) * (I * nxyz*nxyz - xyz * xyz.transpose())/(nxyz*nxyz*nxyz);
        J = J * negative * Qmat;
      }
    }
    return true;
  }

private:
  quat::Quat<double> quat_;
};

class QuatFactorCostFunction
{
public:
  QuatFactorCostFunction(double *x)
  {
    q_ = Quatd(x);
  }

  template<typename T>
  bool operator()(const T* _q2, T* res) const
  {
    quat::Quat<T> q2(_q2);
    Map<Matrix<T,3,1>> r(res);
    r = q_ - q2;
    return true;
  }

private:
  quat::Quat<double> q_;
};
typedef ceres::AutoDiffCostFunction<QuatFactorCostFunction, 3, 4> QuatFactorAutoDiff;
