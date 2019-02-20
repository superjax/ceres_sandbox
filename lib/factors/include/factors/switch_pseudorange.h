#pragma once
#include <ceres/ceres.h>

#include "factors/pseudorange.h"

class SwitchPRangeFunctor : public PRangeFunctor
{
public:
    SwitchPRangeFunctor(const GTime& _t, const Vector2d& _rho, const Satellite& sat,
                        const Vector3d& _rec_pos_ecef, const Matrix2d& cov, const double& s0,
                        const double& s_weight)
        : PRangeFunctor(_t, _rho, sat, _rec_pos_ecef, cov)
    {
        s0_ = s0;
        sw_ = s_weight;
    }

    void reset(double s0)
    {
        s0_ = s0;
    }

    template <typename T>
    bool operator()(const T* _x, const T* _v, const T* _clk, const T* _x_e2n, const T* _s, T* _res) const
    {
        bool result = PRangeFunctor::operator ()(_x, _v, _clk, _x_e2n, _res);
        T s = *_s;
        if (s < 0.0)
            s = (T)0.0;
        else if (s > 1.0)
            s = (T)1.0;

        _res[0] *= s;
        _res[1] *= s;
        _res[2] = sw_ * (s0_ - s);
        return result;
    }

    double s0_;
    double sw_;
};

typedef ceres::AutoDiffCostFunction<SwitchPRangeFunctor, 3, 7, 3, 2, 7, 1> SwitchPRangeFactorAD;


class SwitchDynamicsFunctor
{
public:
    SwitchDynamicsFunctor(const double& weight)
    {
        sw_ = weight;
    }

    template <typename T>
    bool operator()(const T* _si, const T* _sj, T* _res) const
    {
        *_res = sw_ * (*_si - *_sj);
        return true;
    }

    double sw_;
};
typedef ceres::AutoDiffCostFunction<SwitchDynamicsFunctor, 1, 1, 1> SwitchDynamicsFactorAD;
