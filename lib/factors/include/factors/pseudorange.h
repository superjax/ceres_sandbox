#pragma once

#include <ceres/ceres.h>
#include <Eigen/Core>

#include "multirotor_sim/satellite.h"
#include "multirotor_sim/wsg84.h"


using namespace Eigen;

class PseudorangeCostFunction
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    PseudorangeCostFunction(const GTime& _t, const Vector2d& _rho, Satellite& sat, const Vector3d& _rec_pos_ecef, const Matrix2d& cov)
    {
        // We don't have ephemeris for this satellite, we can't do anything with it yet
        if (sat.eph_.A == 0)
            return;

        t = _t;
        rho = _rho;
        rec_pos = _rec_pos_ecef;
        sat.computePositionVelocityClock(t, sat_pos, sat_vel, sat_clk_bias);

        // Earth rotation correction. The change in velocity can be neglected.
        Vector3d los_to_sat = sat_pos - rec_pos;
        double tau = los_to_sat.norm() / Satellite::C_LIGHT;
        sat_pos -= sat_vel * tau;
        double xrot = sat_pos.x() + sat_pos.y() * Satellite::OMEGA_EARTH * tau;
        double yrot = sat_pos.y() - sat_pos.x() * Satellite::OMEGA_EARTH * tau;
        sat_pos.x() = xrot;
        sat_pos.y() = yrot;

        los_to_sat = sat_pos - rec_pos;
        Vector2d az_el = sat.los2azimuthElevation(rec_pos, los_to_sat);
        ion_delay = sat.ionosphericDelay(t, WSG84::ecef2lla(rec_pos), az_el);
        Xi_ = cov.inverse().llt().matrixL().transpose();
        valid = true;
    }

    template <typename T>
    bool operator()(const T* _x, const T* _v, const T* _clk, const T* _x_e2n, T* _res)
    {
        typedef Matrix<T,3,1> Vec3;
        typedef Matrix<T,2,1> Vec2;


        Xform<T> x(_x);
        Map<const Vec3> v_NED(_v);
        Map<const Vec2> clk(_clk);
        Xform<T> x_e2n(_x_e2n);
        Map<Vec2> res(_res);


        Vec3 v_ECEF = x.q().rota(v_NED);
        Vec3 p_ECEF = x_e2n.transforma(x.t());
        Vec3 los_to_sat = sat_pos - p_ECEF;

        Vec2 rho_hat;
        rho_hat(0) = los_to_sat.norm() + ion_delay - (T)Satellite::C_LIGHT*(sat_clk_bias(0) + clk(0));
        rho_hat(1) = (sat_vel - v_ECEF).dot(los_to_sat / rho_hat(0)) - (T)Satellite::C_LIGHT*(sat_clk_bias(1) + clk(1));

        res = rho - rho_hat;

        res = Xi_ * res;

        /// TODO: Check if time or rec_pos have deviated too much and re-calculate ion_delay and earth rotation effect

        return true;
    }

    bool valid = false;
    GTime t;
    Vector2d rho;
    Vector3d sat_pos;
    Vector3d sat_vel;
    Vector2d sat_clk_bias;
    double ion_delay;
    Vector3d rec_pos;
    Matrix2d Xi_;
};

typedef ceres::AutoDiffCostFunction<PseudorangeCostFunction, 2, 7, 3, 2, 7> PRangeAD;
