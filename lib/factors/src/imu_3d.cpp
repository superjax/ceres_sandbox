#include <ceres/ceres.h>
#include <Eigen/Core>

#include "geometry/xform.h"

#include "factors/imu_3d.h"

using namespace Eigen;
using namespace xform;


Imu3DFactorCostFunction::Imu3DFactorCostFunction(double _t0, const Vector6d& bhat, const Matrix6d& cov)
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

void Imu3DFactorCostFunction::dynamics(const Vector10d& y, const Vector6d& u,
                                       Vector9d& ydot, Matrix9d& A, Matrix<double, 9, 6>&B, Matrix<double, 9, 6>& C)
{
    Map<const Vector3d> alpha(y.data()+ALPHA);
    Map<const Vector3d> beta(y.data()+BETA);
    Quatd gamma(y.data()+GAMMA);
    Map<const Vector3d> a(u.data()+ACC);
    Map<const Vector3d> w(u.data()+OMEGA);
    Map<Vector3d> ba(bhat_.data()+ACC);
    Map<Vector3d> bw(bhat_.data()+OMEGA);

    // ydot = Ay + Bu + Cb

    ydot.segment<3>(ALPHA) = beta;
    ydot.segment<3>(BETA) = gamma.rota(a - ba);
    ydot.segment<3>(GAMMA) = w - bw;

    A.setZero();
    A.block<3,3>(ALPHA, BETA) = I_3x3;
    A.block<3,3>(BETA, GAMMA) = -gamma.R().transpose() * skew(a - ba);

    B.setZero();
    B.block<3,3>(BETA, ACC) = gamma.R().transpose();
    B.block<3,3>(GAMMA, OMEGA) = I_3x3;

    C = -B;
}

void Imu3DFactorCostFunction::boxplus(const Vector10d& y, const Vector9d& dy, Vector10d& yp)
{
    yp.segment<3>(P) = y.segment<3>(P) + dy.segment<3>(P);
    yp.segment<3>(V) = y.segment<3>(V) + dy.segment<3>(V);
    yp.segment<4>(Q) = (Quatd(y.segment<4>(Q)) + dy.segment<3>(Q)).elements();
}

void Imu3DFactorCostFunction::boxminus(const Vector10d& y1, const Vector10d& y2, Vector9d& d)
{

    d.segment<3>(P) = y1.segment<3>(P) - y2.segment<3>(P);
    d.segment<3>(V) = y1.segment<3>(V) - y2.segment<3>(V);
    d.segment<3>(Q) = Quatd(y1.segment<4>(Q)) - Quatd(y2.segment<4>(Q));
}

void Imu3DFactorCostFunction::integrate(double _t, const Vector6d& u, bool record)
{
    double dt = _t - (t0_ + delta_t_);
    delta_t_ = _t - t0_;

    if (record)
    {
        imu_hist_.push_back(u);
        t_hist_.push_back(_t);
    }

    Vector9d ydot;
    Matrix9d A;
    Matrix<double, 9, 6> B, C;
    Vector10d yp;
    dynamics(y_, u, ydot, A, B, C);
    boxplus(y_, ydot * dt, yp);
    y_ = yp;

    A = Matrix9d::Identity() + A*dt + 1/2.0 * A*A*dt*dt;
    B = B*dt;
    C = C*dt;

    P_ = A*P_*A.transpose() + B*imu_cov_*B.transpose();
    J_ = A*J_ + C;
}

void Imu3DFactorCostFunction::estimate_xj(const double* _xi, const double* _vi, double* _xj, double* _vj) const
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

void Imu3DFactorCostFunction::finished()
{
    Omega_ = P_.inverse();
}
