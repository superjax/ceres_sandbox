#pragma once

#include <ceres/ceres.h>
#include <Eigen/Core>

using namespace Eigen;

class Pose1DConstraint : public ceres::SizedCostFunction<2,2,3>
{
public:
  Pose1DConstraint(Matrix2d cov)
  {
    Omega_ = cov.inverse();
  }

  virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const
  {
    Map<const Vector2d> xi(parameters[0]);
    Map<const Vector3d> xj(parameters[1]);
    Map<Vector2d> r(residuals);
    r = xi - xj.block<2,1>(0,0);
    r = Omega_ * r;

    if (jacobians)
    {
      if (jacobians[0])
      {
        jacobians[0][0] = 1; jacobians[0][1] = 0;
        jacobians[0][2] = 0; jacobians[0][3] = 1;
      }
      if (jacobians[1])
      {
        jacobians[1][0] = -1; jacobians[1][1] = 0;  jacobians[1][2] = 0;
        jacobians[1][3] = 0;  jacobians[1][4] = -1; jacobians[1][5] = 0;
      }
    }
    return true;
  }

protected:
  Matrix2d Omega_;
};

class Imu1DFactorCostFunction
{
public:
    Imu1DFactorCostFunction(double _t0, double _bi_hat, Matrix2d _Q)
    {
      t0_ = _t0;
      delta_t_ = 0;
      bi_hat_ = _bi_hat;
      y_.setZero();
      P_.setZero();
      Q_ = _Q;
      J_ = Matrix3d::Identity();
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

      // propagate Jacobian dy/dxi
      J_ = (IpAdt)*J_;
    }

    Vector3d estimate_xj(const Vector3d& xi) const
    {
      // Integrate starting at origin pose to get a measurement of the final pose
      Vector3d xj;
      xj(P) = xi(P) + xi(V)*delta_t_ + y_(ALPHA);
      xj(V) = xi(V) + y_(BETA);
      xj(B) = xi(B) + y_(DB);
      return xj;
    }

    void finished()
    {
      Omega_ = P_.inverse();
    }

    template<typename T>
    bool operator()(const T* _xi, const T* _xj, T *residuals) const
    {
      typedef Matrix<T, 3, 1> Vec3;
      Map<const Vec3> xi(_xi);
      Map<const Vec3> xj(_xj);
      Map<Vec3> r(residuals);

      // Use the jacobian to re-calculate y_ with change in bias
      T db = xi(B) - bi_hat_;

      Vec3 y_db = y_ + J_.col(DB) * db;

      r(P) = (xj(P) - xi(P) - xi(V)*delta_t_) - y_db(ALPHA);
      r(V) = (xj(V) - xi(V)) - y_db(BETA);
      r(B) = (xj(B) - xi(B)) - y_db(DB);
      r = Omega_ * r;

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
    Matrix3d J_;

    const Matrix3d A_ = (Matrix3d()
                         << 0, 1, 0,
                            0, 0, -1,
                            0, 0, 0).finished();
    const Matrix<double, 3, 2> B_ = (Matrix<double, 3, 2>()
                                     << 0, 0,
                                       -1, 0,
                                        0, 1).finished();
};
typedef ceres::AutoDiffCostFunction<Imu1DFactorCostFunction, 3, 3, 3> Imu1DFactorAutoDiff;



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
      J_ = Matrix3d::Identity();
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

      // propagate Jacobian dy/dxi
      J_ = (IpAdt)*J_;
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

      // Use the jacobian to re-calculate y_ with change in bias
      double db = xi(B) - bi_hat_;

      Vector3d y_db = y_ + J_.col(DB) * db;

      r(P) = (xj(P) - xi(P) - xi(V)*delta_t_) - y_db(ALPHA);
      r(V) = (xj(V) - xi(V)) - y_db(BETA);
      r(B) = (xj(B) - xi(B)) - y_db(DB);

//      r = Omega_ * r;

      if (jacobians)
      {
        if (jacobians[0])
        {
          Map<Matrix3d, RowMajor> drdxi(jacobians[0]);
          drdxi(P,P) = -1;    drdxi(P,V) = -delta_t_;    drdxi(P,B) = -J_(ALPHA, DB);
          drdxi(V,P) = 0;     drdxi(V,V) = -1;           drdxi(V,B) = -J_(BETA, DB);
          drdxi(B,P) = 0;     drdxi(B,V) = 0;            drdxi(B,B) = -1 - J_(B, DB);
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
    Matrix3d J_;

    const Matrix3d A_ = (Matrix3d()
                         << 0, 1, 0,
                            0, 0, -1,
                            0, 0, 0).finished();
    const Matrix<double, 3, 2> B_ = (Matrix<double, 3, 2>()
                                     << 0, 0,
                                       -1, 0,
                                        0, 1).finished();
};
