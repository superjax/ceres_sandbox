#pragma once

#include <ceres/ceres.h>
#include <Eigen/Core>

using namespace Eigen;

class Imu1DFactor : public ceres::SizedCostFunction<3, 3, 3>
{
public:
    Imu1DFactor(double _t0, double _bi_hat, Matrix2d _Q)
    {
      t0_ = _t0;
      delta_t_ = 0;
      bi_hat_ = _bi_hat;
      y_.setZero();
      P_.setZero();
      Q_ = _Q;
    }

    void integrate(double _t, double a)
    {
      double dt = _t - (t0_ + delta_t_);
      delta_t_ = _t - t0_;

      // propagate state
      y_(ALPHA) = y_(ALPHA) + y_(BETA,0)*dt + 0.5*(a - bi_hat_)*dt*dt;
      y_(BETA) = y_(BETA) + (a - bi_hat_)*dt;

      // propagate covariance
      Matrix3d IpAdt = Matrix3d::Identity() + A_ * dt;
      Matrix<double, 3, 2> Bdt = B_ * dt;
      P_ = IpAdt * P_ * IpAdt.transpose() + Bdt * Q_ * Bdt.transpose();
    }

    Vector3d estimate_xj(const Vector3d& xi) const
    {
      // Integrate starting at origin pose to get a measurement of the final pose
      Vector3d xj;
      xj(P) = xi(P) + 0.5 * xi(V)*delta_t_ + y_(ALPHA);
      xj(V) = xi(V) + y_(BETA);
      xj(B) = xi(B) + y_(DB);
      return xj;
    }

    void finished()
    {
      Omega_ = P_.inverse();
    }

    virtual bool Evaluate(const double * const *parameters, double *residuals, double **jacobians) const
    {
      Map<const Vector3d> xi(parameters[0]);
      Map<const Vector3d> xj(parameters[1]);
      Map<Vector3d> r(residuals);

      r(P) = (xj(P) - xi(P) - xi(V)*delta_t_) - y_(ALPHA);
      r(V) = (xj(V) - xi(V)) - y_(BETA);
      r(B) = (xj(B) - xi(B)) - y_(DB);

      r = Omega_ * r;

      if (jacobians)
      {
        if (jacobians[0])
        {
          Map<Matrix3d, RowMajor> drdxi(jacobians[0]);
          drdxi(P,P) = -1;    drdxi(P,V) = -delta_t_;    drdxi(P,B) = 0;
          drdxi(V,P) = 0;     drdxi(V,V) = -1;           drdxi(V,B) = 0;
          drdxi(B,P) = 0;     drdxi(B,V) = 0;            drdxi(B,B) = -1;
        }
        if (jacobians[1])
        {
          Map<Matrix3d, RowMajor> drdxj(jacobians[1]);
          drdxj(P,P) = 1;     drdxj(P,V) = 0;           drdxj(P,B) = 0;
          drdxj(V,P) = 0;     drdxj(V,V) = 1;           drdxj(V,B) = 0;
          drdxj(B,P) = 0;     drdxj(B,V) = 0;           drdxj(B,B) = 1;
        }
      }
      return true;
    }

private:

    enum {
      ALPHA = 0,
      BETA = 1,
      DB = 2
    };
    enum {
      P = 0,
      V = 1,
      B = 2
    };

    double t0_;
    double bi_hat_;
    double delta_t_;
    Matrix3d P_;
    Matrix3d Omega_;
    Vector3d y_;
    Matrix2d Q_;

    const Matrix3d A_ = (Matrix3d()
                         << 0, 1, 0,
                            0, 0, 0,
                            0, 0, 0).finished();
    const Matrix<double, 3, 2> B_ = (Matrix<double, 3, 2>()
                                     << 0, 0,
                                       -1, 0,
                                        0, 1).finished();
};
