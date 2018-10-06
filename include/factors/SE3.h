#pragma once

#include <ceres/ceres.h>
#include "factors/attitude_3d.h"
#include "geometry/xform.h"

using namespace Eigen;
using namespace xform;

class XformFactorCostFunction
{
public:
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
    xform::Xform<double> xform_;
};
typedef ceres::AutoDiffCostFunction<XformFactorCostFunction, 6, 7> XformFactorAutoDiff;



class XformEdgeFactorCostFunction
{
public:
    XformEdgeFactorCostFunction(double *_ebar_ij, double *_P_ij)
    {
      ebar_ij_ = Xformd(_ebar_ij);
      Omega_ij_ = Map<const Matrix6d, RowMajor>(_P_ij).inverse();
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
    XformNodeFactorCostFunction(const double *_xbar, const double *_P)
    {
      xbar_ = Xformd(_xbar);
      Omega_ = Map<const Matrix6d, RowMajor>(_P).inverse();
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

