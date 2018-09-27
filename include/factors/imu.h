#pragma once

#include <ceres/ceres.h>
#include <Eigen/Core>


class Imu1DFactor : public ceres::SizedCostFunction<2, 2, 2, 1>
{
public:
    Imu1DFactor(double _t0)
    {
      t0_ = _t0;
      dv_ = 0;
    }
    void integrate(double _t, double a)
    {
      double dt =  _t - (delta_t_ + t0_);
      delta_t_ = _t - t0_;
      dv_ += a*dt;
    }
    Eigen::Vector2d propagate(const Eigen::Vector2d& xi) const
    {
      // Integrate starting at origin pose to get a measurement of the final pose
      Eigen::Vector2d xj;
      xj(1) = xi(1) + dv_;
      xj(0) = xi(0) + (delta_t_/2.0) * (xj(1) + xi(1));
      return xj;
    }

    virtual bool Evaluate(const double * const *parameters, double *residuals, double **jacobians) const
    {
      // estimated state before interval
      double xhati = parameters[0][0];
      double vhati = parameters[0][1];

      // estimated state after interval
      double xhatj = parameters[1][0];
      double vhatj = parameters[1][1];

      // Bias over interval
      double b_ij = parameters[2][0];

      // Calculate residual
      residuals[0] = xhatj - (xhati + (delta_t_/2.0) * (vhatj + vhati));
      residuals[1] = vhatj - (vhati + dv_ + (b_ij * delta_t_));

      if (jacobians)
      {
        if (jacobians[0])
        {
          // dr/dxhati          dr/dvhati
          jacobians[0][0] = -1; jacobians[0][1] = -delta_t_/2.0;
          jacobians[0][2] = 0;  jacobians[0][3] = -1;
        }
        if (jacobians[1])
        {
          // dr/dxhatj          dr/dvhatj
          jacobians[1][0] = 1;  jacobians[1][1] = -delta_t_/2.0;
          jacobians[1][2] = 0;  jacobians[1][3] = 1;
        }
        if (jacobians[2])
        {
          // dr/db_ij
          jacobians[2][0] = 0;
          jacobians[2][1] = -delta_t_;
        }
      }
      return true;
    }

private:
    double t0_;
    double dv_;
    double delta_t_;
};
