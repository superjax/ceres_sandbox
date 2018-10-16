#pragma once

#include <ceres/ceres.h>
#include <Eigen/Core>

#include "geometry/xform.h"

using namespace Eigen;
using namespace xform;

class Imu3DFactorCostFunction
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Imu3DFactorCostFunction(double _t0, const Vector6d& bhat, const Matrix6d& cov);
    void dynamics(const Vector10d& y, const Vector6d& u,
                  Vector9d& ydot, Matrix9d& A, Matrix<double, 9, 6>&B, Matrix<double, 9, 6>& C);
    void boxplus(const Vector10d& y, const Vector9d& dy, Vector10d& yp);
    void boxminus(const Vector10d& y1, const Vector10d& y2, Vector9d& d);
    void integrate(double _t, const Vector6d& u, bool record=true);
    void estimate_xj(const double* _xi, const double* _vi, double* _xj, double* _vj) const;
    void finished();

    template<typename T>
    bool operator()(const T* _xi, const T* _xj, const T* _vi, const T* _vj, const T* _b, T* residuals) const
    {
        typedef Matrix<T,3,1> Vec3;
        typedef Matrix<T,6,1> Vec6;
        typedef Matrix<T,9,1> Vec9;
        typedef Matrix<T,10,1> Vec10;

        Xform<T> xi(_xi);
        Xform<T> xj(_xj);
        Map<const Vec3> vi(_vi);
        Map<const Vec3> vj(_vj);
        Map<const Vec6> b(_b);

        Vec9 dy = J_ * (b - bhat_);
        Vec10 y;
        y.block(0,0,6,1) = y_.block(0,0,6,1) + dy.block(0,0,6,1);
        y.block(6,0,4,1) = (Quatd(y_.block(6,0,4,1)).otimes2(Quat<T>::exp(dy.block(6,0,3,1)))).elements();

        Map<Vec3> alpha(y.data()+ALPHA);
        Map<Vec3> beta(y.data()+BETA);
        Quat<T> gamma(y.data()+GAMMA);
        Map<Matrix<T, 9, 1>> r(residuals);

        r.template block<3,1>(ALPHA, 0) = xi.q_.rotp(xj.t_ - xi.t_ - 1/2.0*gravity_*delta_t_*delta_t_) - vi*delta_t_ - alpha;
        r.template block<3,1>(BETA, 0) = gamma.rota(vj) - vi - xi.q_.rotp(gravity_)*delta_t_ - beta;
        r.template block<3,1>(GAMMA, 0) = (xi.q_.inverse() * xj.q_) - gamma;

//        r = Omega_ * r;

        return true;
    }

    enum : int
    {
        ALPHA = 0,
        BETA = 3,
        GAMMA = 6,
    };

    enum :int
    {
        ACC = 0,
        OMEGA = 3
    };

    enum : int
    {
        P = 0,
        V = 3,
        Q = 6,
    };

    double t0_;
    double delta_t_;

    Matrix9d P_;
    Matrix9d Omega_;
    Vector6d bhat_;
    Vector10d y_;

    Matrix6d imu_cov_;
    Matrix<double, 9, 6> J_;
    Vector3d gravity_ = (Vector3d() << 0, 0, 9.80665).finished();
    std::vector<Vector6d, aligned_allocator<Vector6d>> imu_hist_;
    std::vector<double> t_hist_;
};

typedef ceres::AutoDiffCostFunction<Imu3DFactorCostFunction, 9, 7, 7, 3, 3, 6> Imu3DFactorAutoDiff;
