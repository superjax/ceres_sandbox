#pragma once

#include <ceres/ceres.h>
#include <Eigen/Dense>


class Pos3DFactor : public ceres::SizedCostFunction<3,3>
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Pos3DFactor(double *x)
  {
      position_[0] = x[0];
      position_[1] = x[1];
      position_[2] = x[2];
  }

  virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const
  {
    Eigen::Map<Eigen::Vector3d> res(residuals);
    Eigen::Map<const Eigen::Vector3d> z(parameters[0]);
    res = position_ - z;

    if (jacobians)
    {
      if (jacobians[0])
      {
        Eigen::Map<Eigen::Matrix3d, Eigen::RowMajor> drdz(jacobians[0]);
        drdz = -1.0 * Eigen::Matrix3d::Identity();
      }
    }
    return true;
  }

 protected:
  Eigen::Vector3d position_;
};
