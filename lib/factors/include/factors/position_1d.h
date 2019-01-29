#pragma once


#include <ceres/ceres.h>
#include <Eigen/Dense>


class Pos1DFactor : public ceres::SizedCostFunction<1,1>
{
public:
  Pos1DFactor(double x) :
    position(x)
  {}

  virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const
  {
    residuals[0] = position - parameters[0][0];

    if (jacobians)
    {
      if (jacobians[0])
        jacobians[0][0] = -1;
    }
    return true;
  }

protected:
  double position;
};

class Pos1DTimeOffsetFactor : public ceres::SizedCostFunction<1,2,1>
{
public:
  Pos1DTimeOffsetFactor(double x, double cov) :
    xbar_(x), cov_(cov)
  {}

  virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const
  {
    double xhat = parameters[0][0];
    double zhat = parameters[0][1];
    double Tdhat = parameters[1][0];

    residuals[0] = 1.0/cov_*((xhat + zhat*Tdhat) - xbar_);

    if (jacobians)
    {
      if (jacobians[0])
      {
        jacobians[0][0] = 1.0/cov_;
        jacobians[0][1] = Tdhat/cov_;
      }
      if (jacobians[1])
      {
        jacobians[1][0] = zhat/cov_;
      }
    }
    return true;
  }

protected:
  double xbar_;
  double cov_;
};

