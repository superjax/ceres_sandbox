#include <fstream>

#include <ceres/ceres.h>
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <gtest/gtest.h>

#include "geometry/xform.h"
#include "multirotor_sim/simulator.h"
#include "multirotor_sim/controller.h"
#include "multirotor_sim/satellite.h"
#include "multirotor_sim/estimator_wrapper.h"
#include "utils/logger.h"
#include "test_common.h"
#include "utils/trajectory_sim.h"

#include "factors/pseudorange.h"
#include "factors/SE3.h"
#include "factors/clock_bias_dynamics.h"
#include "factors/carrier_phase.h"
#include "factors/imu_3d.h"

class TestCarrierPhase : public ::testing::Test
{
protected:
    TestCarrierPhase() :
        sat(1, 0)
    {}
    void SetUp() override
    {
        time.week = 86400.00 / DateTime::SECONDS_IN_WEEK;
        time.tow_sec = 86400.00 - (time.week * DateTime::SECONDS_IN_WEEK);

        eph.sat = 1;
        eph.A = 5153.79589081 * 5153.79589081;
        eph.toe.week = 93600.0 / DateTime::SECONDS_IN_WEEK;
        eph.toe.tow_sec = 93600.0 - (eph.toe.week * DateTime::SECONDS_IN_WEEK);
        eph.toes = 93600.0;
        eph.deln =  0.465376527657e-08;
        eph.M0 =  1.05827953357;
        eph.e =  0.00223578442819;
        eph.omg =  2.06374037770;
        eph.cus =  0.177137553692e-05;
        eph.cuc =  0.457651913166e-05;
        eph.crs =  88.6875000000;
        eph.crc =  344.96875;
        eph.cis = -0.856816768646e-07;
        eph.cic =  0.651925802231e-07;
        eph.idot =  0.342514267094e-09;
        eph.i0 =  0.961685061380;
        eph.OMG0 =  1.64046615454;
        eph.OMGd = -0.856928551657e-08;
        sat.addEphemeris(eph);

    }
    eph_t eph;
    GTime time;
    Satellite sat;
};

TEST_F(TestCarrierPhase, CheckResidualAtInit)
{
    Vector3d provo_lla{40.246184 * DEG2RAD , -111.647769 * DEG2RAD, 1387.997511};
    Vector3d rec_pos = WSG84::lla2ecef(provo_lla);
    Xformd x_e2n = WSG84::x_ecef2ned(rec_pos);

    double dt = 2.5;

    GTime t0 = time;
    GTime t1 = t0 + dt;

    Vector3d z0, z1;
    sat.computeMeasurement(t0, rec_pos, Vector3d::Zero(), Vector2d::Zero(), z0);
    sat.computeMeasurement(t1, rec_pos, Vector3d::Zero(), Vector2d::Zero(), z1);

    double dPhi = z1(2) - z0(2);
    CarrierPhaseFunctor phase_factor(t0, t1, dPhi, sat, rec_pos, 1.0);

    Xformd x = Xformd::Identity();
    Vector2d clk = Vector2d::Zero();
    double res = NAN;

    phase_factor(x.data(), x.data(), clk.data(), clk.data(), x_e2n.data(), &res);

    EXPECT_NEAR(res, 0.0, 1e-2);
}

TEST_F(TestCarrierPhase, CheckResidualAfterMovingAtInit)
{
    Vector3d provo_lla{40.246184 * DEG2RAD , -111.647769 * DEG2RAD, 1387.997511};
    Vector3d rec_pos = WSG84::lla2ecef(provo_lla);
    Xformd x_e2n = WSG84::x_ecef2ned(rec_pos);

    Xformd x0 = Xformd::Identity();
    Xformd x1 = Xformd::Identity();
    x1.t() += Vector3d{10, 0, 0};

    double dt = 2.5;

    GTime t0 = time;
    GTime t1 = t0 + dt;

    Vector3d z0, z1;
    sat.computeMeasurement(t0, WSG84::ned2ecef(x_e2n, x0.t()), Vector3d::Zero(), Vector2d::Zero(), z0);
    sat.computeMeasurement(t1, WSG84::ned2ecef(x_e2n, x1.t()), Vector3d::Zero(), Vector2d::Zero(), z1);

    double dPhi = z1(2) - z0(2);
    CarrierPhaseFunctor phase_factor(t0, t1, dPhi, sat, rec_pos, 1.0);

    Vector2d clk = Vector2d::Zero();
    double res = NAN;

    phase_factor(x0.data(), x1.data(), clk.data(), clk.data(), x_e2n.data(), &res);

    EXPECT_NEAR(res, 0.0, 1e-2);
}

TEST (CarrierPhase, Trajectory)
{
    TrajectorySim a(raw_gps_yaml_file());

    a.addNodeCB([&a](){a.initNodePostionFromPointPos();});
    a.addNodeCB([&a](){a.addPseudorangeFactorsWithClockDynamics();});
    a.addNodeCB([&a](){a.addCarrierPhaseFactors();});

    a.run();
    a.solve();
    a.log("/tmp/ceres_sandbox/CarrierPhase.Trajectory.log");

    ASSERT_LE(a.final_error(), a.error0);
}

TEST (CarrierPhase, ImuTrajectory)
{
    TrajectorySim a(raw_gps_yaml_file());
    a.fix_origin();

    a.addNodeCB([&a](){a.addImuFactor();});
    a.addNodeCB([&a](){a.initNodePostionFromPointPos();});
    a.addNodeCB([&a](){a.addPseudorangeFactorsWithClockDynamics();});
    a.addNodeCB([&a](){a.addCarrierPhaseFactors();});

    a.run();
    a.solve();
    a.log("/tmp/ceres_sandbox/CarrierPhase.ImuTrajectory.log");

    ASSERT_LE(a.final_error(), a.error0);
}


