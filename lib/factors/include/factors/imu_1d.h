#pragma once

#include <ceres/ceres.h>
#include <Eigen/Core>

using namespace Eigen;

class Pose1DFactor : public ceres::SizedCostFunction<2,2>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Pose1DFactor(Vector2d z, Matrix2d cov)
  {
    z_ = z;
    Omega_ = cov.inverse();
  }

  virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const
  {
    Map<const Vector2d> x(parameters[0]);
    Map<Vector2d> r(residuals);
    r = x - z_;
    r = Omega_ * r;

    if (jacobians)
    {
      if (jacobians[0])
      {
        Map<Matrix2d> J(jacobians[0]);
        J = Omega_;
      }
    }
    return true;
  }

protected:
  Vector2d z_;
  Matrix2d Omega_;
};

class Imu1DFunctor
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Imu1DFunctor(double _t0, double _bi_hat, double avar)
    {
      t0_ = _t0;
      delta_t_ = 0;
      bi_hat_ = _bi_hat;
      y_.setZero();
      P_.setZero();
      avar_ = avar;
      J_.setZero();
    }

    void integrate(double _t, double a)
    {
      double dt = _t - (t0_ + delta_t_);
      delta_t_ = _t - t0_;

      // propagate covariance
      Matrix2d A = (Matrix2d() << 1.0, dt, 0.0, 1.0).finished();
      Vector2d B {0.5*dt*dt, dt};
      Vector2d C {-0.5*dt*dt, dt};

      // Propagate state
      y_ = A*y_ + B*a + C*bi_hat_;

      P_ = A*P_*A.transpose() + B*avar_*B.transpose();

      // propagate Jacobian dy/db
      J_ = A*J_ + C;
    }

    Vector2d estimate_xj(const Vector2d& xi) const
    {
      // Integrate starting at origin pose to get a measurement of the final pose
      Vector2d xj;
      xj(P) = xi(P) + xi(V)*delta_t_ + y_(ALPHA);
      xj(V) = xi(V) + y_(BETA);
      return xj;
    }

    void finished()
    {
      Omega_ = P_.inverse();
    }

    template<typename T>
    bool operator()(const T* _xi, const T* _xj, const T* _b, T *residuals) const
    {
      typedef Matrix<T, 2, 1> Vec2;
      Map<const Vec2> xi(_xi);
      Map<const Vec2> xj(_xj);
      Map<Vec2> r(residuals);

      // Use the jacobian to re-calculate y_ with change in bias
      T db = *_b - bi_hat_;

      Vec2 y_db = y_ + J_ * db;

      r(P) = (xj(P) - xi(P) - xi(V)*delta_t_) - y_db(ALPHA);
      r(V) = (xj(V) - xi(V)) - y_db(BETA);
      r = Omega_ * r;

      return true;
    }
private:

    enum {
      ALPHA = 0,
      BETA = 1,
    };
    enum {
      P = 0,
      V = 1
    };

    double t0_;
    double bi_hat_;
    double delta_t_;
    Matrix2d P_;
    Matrix2d Omega_;
    Vector2d y_;
    double avar_;
    Vector2d J_;
};
typedef ceres::AutoDiffCostFunction<Imu1DFunctor, 2, 2, 2, 1> Imu1DFactorAD;
