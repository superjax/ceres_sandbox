#pragma once

#include <ceres/ceres.h>
#include <Eigen/Core>

#include "multirotor_sim/satellite.h"

using namespace Eigen;

class ClockBiasFunctor
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    ClockBiasFunctor(const double& dt, const Matrix2d& cov)
    {
        dt_ = dt;
        Xi_ = cov.inverse().llt().matrixL().transpose();
    }

    template <typename T>
    bool operator()(const T* _taui, const T* _tauj, T* _res) const
    {
        typedef Matrix<T,2,1> Vec2;

        Map<const Vec2> tau_i(_taui);
        Map<const Vec2> tau_j(_tauj);
        Map<Vec2> res(_res);

        res(0) = (tau_i(0) + tau_i(1) * (T)dt_) - tau_j(0);
        res(1) = (tau_i(1)) - tau_j(1);

        res = Xi_ * res;
        return true;
    }

    double dt_;
    Matrix2d Xi_;
};
typedef ceres::AutoDiffCostFunction<ClockBiasFunctor, 2, 2, 2> ClockBiasFactorAD;
