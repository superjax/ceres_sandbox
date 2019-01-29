#pragma once

#include <ceres/ceres.h>
#include <Eigen/Core>

#include "multirotor_sim/satellite.h"
#include "multirotor_sim/wsg84.h"

#include "geometry/xform.h"


using namespace Eigen;
using namespace xform;

class CarrierPhaseFunctor
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    struct SatState
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        GTime t;
        Vector3d pos;
        Vector3d vel;
        Vector2d tau;
    };


    CarrierPhaseFunctor(const GTime& _t0, const GTime& _t1, const double& _dPhi, const Satellite& sat, const Vector3d& _rec_pos_ecef, const double& var)
    {
        // We don't have ephemeris for this satellite, we can't do anything with it yet
        if (sat.eph_.A == 0)
            return;

        sats[0].t = _t0;
        sats[1].t = _t1;
        dPhi = _dPhi;
        rec_pos = _rec_pos_ecef;
        Xi_ = std::sqrt(1.0/var);

        for (int i = 0; i < 2; i++)
        {
            sat.computePositionVelocityClock(sats[i].t, sats[i].pos, sats[i].vel, sats[i].tau);

            // Earth rotation correction. The change in velocity can be neglected.
            Vector3d los_to_sat = sats[i].pos - rec_pos;
            double tof = los_to_sat.norm() / Satellite::C_LIGHT;
            sats[i].pos -= sats[i].vel * tof;
            double xrot = sats[i].pos.x() + sats[i].pos.y() * Satellite::OMEGA_EARTH * tof;
            double yrot = sats[i].pos.y() - sats[i].pos.x() * Satellite::OMEGA_EARTH * tof;
            sats[i].pos.x() = xrot;
            sats[i].pos.y() = yrot;
        }
        valid = true;
    }

    template <typename T>
    bool operator() (const T* _x0, const T* _x1, const T* _tau0, const T* _tau1, const T* _x_e2n, T* res) const
    {
        typedef Matrix<T,2,1> Vec2;
        typedef Matrix<T,3,1> Vec3;

        Xform<T> x0(_x0);
        Xform<T> x1(_x1);
        Xform<T> x_e2n(_x_e2n);
        Map<const Vec2> tau0(_tau0);
        Map<const Vec2> tau1(_tau1);

        Vec3 p0_ECEF = x_e2n.transforma(x0.t());
        Vec3 p1_ECEF = x_e2n.transforma(x1.t());

        T dPhi_hat = (T)(1.0/Satellite::LAMBDA_L1) * (((sats[1].pos - p1_ECEF).norm() + (T)Satellite::C_LIGHT*(tau1[0] - sats[1].tau[0]))
                                                 - ((sats[0].pos - p0_ECEF).norm() + (T)Satellite::C_LIGHT*(tau0[0] - sats[0].tau[0])));

        *res = Xi_ * (dPhi - dPhi_hat);

        return true;
    }

    SatState sats[2];

    bool valid = false;
    double dPhi;
    Vector3d rec_pos;
    double Xi_;
};

typedef ceres::AutoDiffCostFunction<CarrierPhaseFunctor, 1, 7, 7, 2, 2, 7> CarrierPhaseFactorAD;
