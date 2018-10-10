#pragma once

#include <ceres/ceres.h>
#include <Eigen/Core>
#include <geometry/xform.h>

using namespace Eigen;
using namespace xform;

class Imu3DFactorCostFunction
{
public:
    Imu3DFactorCostFunction(double _t0, const Vector6d& bhat, const Matrix6d& cov)
    {
        delta_t_ = 0;
        t0_ = _t0;
        bhat_ = bhat;

        y_.setZero();
        y_(Q) = 1.0;
        P_.setZero();
        imu_cov_ = cov;
        J_.setZero();
    }

    void dynamics(const Vector10d& y, const Vector6d& u,
                  Vector9d& ydot, Matrix9d& A, Matrix<double, 9, 6>&B, Matrix<double, 9, 6>& C)
    {
        Map<Vector3d> alpha(y_.data()+ALPHA);
        Map<Vector3d> beta(y_.data()+BETA);
        Quatd gamma(y_.data()+GAMMA);
        Map<const Vector3d> a(u.data()+ACC);
        Map<const Vector3d> w(u.data()+OMEGA);
        Map<Vector3d> ba(bhat_.data()+ACC);
        Map<Vector3d> bw(bhat_.data()+OMEGA);

        // ydot = Ay + Bu + Cb

        ydot.segment<3>(ALPHA) = beta;
        ydot.segment<3>(BETA) = gamma.rota(a-ba);
        ydot.segment<3>(GAMMA) = w - bw;

        A.setZero();
        A.block<3,3>(ALPHA, BETA) = I_3x3;
//        A.block<3,3>(ALPHA, GAMMA) = -1/2.0 * gamma.R().transpose() * skew(a-ba) * dt;
        A.block<3,3>(BETA, GAMMA) = -gamma.R().transpose() * skew(a-ba);

        B.setZero();
//        B.block<3,3>(ALPHA, ACC) = 1/2.0 * gamma.R().transpose() * dt;
        B.block<3,3>(BETA, ACC) = gamma.R().transpose();
        B.block<3,3>(GAMMA, OMEGA) = I_3x3;

        C.setZero();
        C = -B;
    }

    void boxplus(const Vector10d& y, const Vector9d& dy, Vector10d& yp)
    {
        yp.segment<3>(P) = y.segment<3>(P) + dy.segment<3>(P);
        yp.segment<3>(V) = y.segment<3>(V) + dy.segment<3>(V);
        yp.segment<4>(Q) = (Quatd(y.segment<4>(Q)) + dy.segment<3>(Q)).elements();
    }

    void integrate(double _t, const Vector6d& u)
    {
        double dt = _t - (t0_ + delta_t_);
        delta_t_ = _t - t0_;

        Vector9d ydot;
        Matrix9d A;
        Matrix<double, 9, 6> B, C;
        Vector10d yp;
        dynamics(y_, u, ydot, A, B, C);
        boxplus(y_, ydot * dt, yp);
        y_ = yp;

        B = B*dt;
        A = Matrix9d::Identity() + A*dt + 1/2.0 * A*A*dt*dt;

        P_ = A*P_*A.transpose() + B*imu_cov_*B.transpose();
        J_ = A*J_ + B;
    }

    void estimate_xj(const double* _xi, const double* _vi, double* _xj, double* _vj) const
    {
        Map<const Vector3d> alpha(y_.data()+ALPHA);
        Map<const Vector3d> beta(y_.data()+BETA);
        Quatd gamma(y_.data()+GAMMA);
        Xformd xi(_xi);
        Xformd xj(_xj);
        Map<const Vector3d> vi(_vi);
        Map<Vector3d> vj(_vj);

        xj.t_ = xi.t_ + vi*delta_t_ + 1/2.0 * gravity_*delta_t_*delta_t_ + xi.q_.rota(alpha);
        vj = vi + xi.q_.rotp(gravity_)*delta_t_ + beta;
        xj.q_ = xi.q_ * gamma;
    }

    void finished()
    {
        Omega_ = P_.inverse();
    }

    template<typename T>
    bool operator()(const T* _xi, const T* _xj, const T* _vi, const T* _vj, const T* _b, T* residuals) const
    {
        typedef Matrix<T,3,1> Vec3;
        typedef Matrix<T,6,1> Vec6;
        typedef Matrix<T,9,1> Vec9;
        typedef Matrix<T,10,1> Vec10;

        Xform<T> Xi(_xi);
        Xform<T> Xj(_xj);
        Map<const Vec3> vi(_vi);
        Map<const Vec3> vj(_vj);
        Map<const Vec6> b(_b);

        Vec9 dy = J_ * (b - bhat_);
        Vec10 y;
        y.block(0,0,6,1) = y_.block(0,0,6,1) + dy.block(0,0,6,1);
        y.block(6,0,4,1) = (Quat<T>::exp(dy.block(6,0,3,1)).inverse() * Quatd(y_.block(6,0,4,1)).inverse()).inverse().elements();

        Map<Vec3> alpha(y.data()+ALPHA);
        Map<Vec3> beta(y.data()+BETA);
        Quat<T> gamma(y.data()+GAMMA);
        Map<Matrix<T, 9, 1>> r(residuals);

        r.block(ALPHA, 0, 3, 1) = (Xi.q_.rotp(Xj.t_ - Xi.t_ + 1/2.0 * gravity_ * delta_t_*delta_t_ - vi*delta_t_)) - alpha;
        r.block(BETA, 0, 3, 1) = (vj + Xi.q_.R() * gravity_*delta_t_ - vi) - beta;
        r.block(GAMMA, 0, 3, 1) = (Xi.q_.inverse() * Xj.q_) - gamma;

//        r = Omega_ * r;
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
};

typedef ceres::AutoDiffCostFunction<Imu3DFactorCostFunction, 9, 7, 7, 3, 3, 6> Imu3DFactorAutoDiff;
