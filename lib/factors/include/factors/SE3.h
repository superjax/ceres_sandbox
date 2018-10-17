#pragma once

#include <ceres/ceres.h>
#include "factors/attitude_3d.h"
#include "geometry/xform.h"

using namespace Eigen;
using namespace xform;

class XformFactorCostFunction
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    XformFactorCostFunction(double *x)
    {
      xform_ = Xformd(x);
    }

    template<typename T>
    bool operator()(const T* _x2, T* res) const
    {
        xform::Xform<T> x2(_x2);
        Map<Matrix<T,6,1>> r(res);
        r = xform_ - x2;
        return true;
    }
private:
    xform::Xformd xform_;
};
typedef ceres::AutoDiffCostFunction<XformFactorCostFunction, 6, 7> XformFactorAutoDiff;



class XformEdgeFactorCostFunction
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    XformEdgeFactorCostFunction(Vector7d& _ebar_ij, Matrix6d& _P_ij)
    {
      ebar_ij_ = _ebar_ij;
      Omega_ij_ = _P_ij.inverse();
    }

    template<typename T>
    bool operator()(const T* _xi, const T* _xj, T* _res) const
    {
      Xform<T> xhat_i(_xi);
      Xform<T> xhat_j(_xj);
      Xform<T> ehat_12 = xhat_i.inverse() * xhat_j;
      Map<Matrix<T,6,1>> res(_res);
      res = Omega_ij_ * (ebar_ij_.boxminus(ehat_12));
      return true;
    }
private:
    xform::Xform<double> ebar_ij_; // Measurement of Edge
    Matrix6d Omega_ij_; // Covariance of measurement
};
typedef ceres::AutoDiffCostFunction<XformEdgeFactorCostFunction, 6, 7, 7> XformEdgeFactorAutoDiff;



class XformNodeFactorCostFunction
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    XformNodeFactorCostFunction(const Vector7d& _xbar, const Matrix6d& _P)
    {
      xbar_ = Xformd(_xbar);
      Omega_ = _P.inverse();
    }

    template<typename T>
    bool operator()(const T* _x, T* _res) const
    {
      Xform<T> xhat(_x);
      Map<Matrix<T,6,1>> res(_res);
      res = Omega_ * (xbar_ - xhat);
      return true;
    }
private:
    xform::Xformd xbar_; // Measurement of Node
    Matrix6d Omega_; // Covariance of measurement
};
typedef ceres::AutoDiffCostFunction<XformNodeFactorCostFunction, 6, 7> XformNodeFactorAutoDiff;




struct XformPlus {
  template<typename T>
  bool operator()(const T* x, const T* delta, T* x_plus_delta) const
  {
    Xform<T> q(x);
    Map<const Matrix<T,6,1>> d(delta);
    Map<Matrix<T,7,1>> qp(x_plus_delta);
    qp = (q + d).elements();
    return true;
  }
};
typedef ceres::AutoDiffLocalParameterization<XformPlus, 7, 6> XformAutoDiffParameterization;


struct XformTimeOffsetCostFunction
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  XformTimeOffsetCostFunction(const Xformd& _x, const Vector6d& _xdot, const Matrix6d& _P)
  {
    Xi_ = _P.inverse().llt().matrixL().transpose();
    xdot_ = _xdot;
    x_ = _x;
  }

  template<typename T>
  bool operator()(const T* _x, const T* _toff, T* _res) const
  {
    typedef Matrix<T,6,1> Vec6;
    Map<Vec6> res(_res);
    Xform<T> x(_x);
    res = Xi_ * (x - (x_.boxplus<T,T>((*_toff) * xdot_)));
    return true;
  }
private:
  Xformd x_;
  Vector6d xdot_;
  Matrix6d Xi_;
};
typedef ceres::AutoDiffCostFunction<XformTimeOffsetCostFunction, 6, 7, 6> XformTimeOffsetAutoDiff;
