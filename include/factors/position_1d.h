#pragma once


#include <ceres/ceres.h>
#include <Eigen/Dense>


class Position1dFactor : public ceres::SizedCostFunction<1,1>
{
 public:
  Position1dFactor(double x) :
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
