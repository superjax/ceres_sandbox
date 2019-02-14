#include <fstream>
#include <random>

#include <ceres/ceres.h>
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <gtest/gtest.h>

#include "geometry/xform.h"
#include "multirotor_sim/simulator.h"
#include "multirotor_sim/controller.h"
#include "multirotor_sim/satellite.h"
#include "utils/estimator_wrapper.h"
#include "utils/logger.h"
#include "test_common.h"

#include "factors/switch.h"
#include "factors/pseudorange.h"


using namespace multirotor_sim;
using namespace Eigen;
using namespace xform;

class TestSwitchPseudorange
{
public:
  TestSwitchPseudorange() :
      sat(1, 0)
  {
      init_sat();
      init_func();
  }

  void init_sat()
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

  void init_func()
  {
      Vector2d rho;
      sat.computeMeasurement(time, rec_pos, Vector3d::Zero(), clk_bias, z);
      rho = z.topRows<2>();
      Matrix2d cov = (Vector2d{3.0, 0.4}).asDiagonal();

      f = new SwitchPRangeFunctor(time, rho, sat, rec_pos, cov, 1.0, 1.0);

      x.t() << 10, 0, 0;
      Vector3d p_ecef = WSG84::ned2ecef(x_e2n, x.t());
      sat.computeMeasurement(time, p_ecef, Vector3d::Zero(), clk_bias, znew);
      true_res << Vector2d{std::sqrt(1/3.0), std::sqrt(1/0.4)}.asDiagonal() * (z - znew).topRows<2>(), 0;
  }

  Vector3d provo_lla{40.246184 * DEG2RAD , -111.647769 * DEG2RAD, 1387.997511};
  Vector3d rec_pos = WSG84::lla2ecef(provo_lla);
  Xformd x_e2n = WSG84::x_ecef2ned(rec_pos);
  Vector2d clk_bias{1e-8, 1e-6};
  Vector3d v = Vector3d::Zero();
  Vector3d res = Vector3d::Zero();
  Xformd x = Xformd::Identity();
  Vector3d z;
  Vector3d znew;
  Vector3d true_res;

  SwitchPRangeFunctor* f;
  eph_t eph;
  GTime time;
  Satellite sat;
};

TEST (SwitchPRangeFunctor, SwitchOn)
{
    TestSwitchPseudorange a;
    double s = 1.0;
    Vector3d res;
    (*a.f)(a.x.data(), a.v.data(), a.clk_bias.data(), a.x_e2n.data(), &s, res.data());

    EXPECT_MAT_NEAR(a.true_res, res, 1e-4);
}

TEST (SwitchPRangeFunctor, SwitchOff)
{
    TestSwitchPseudorange a;
    double s = 0.0;
    Vector3d res;
    (*a.f)(a.x.data(), a.v.data(), a.clk_bias.data(), a.x_e2n.data(), &s, res.data());

    EXPECT_MAT_NEAR(Vector3d(0, 0, 1.0), res, 1e-4);
}

TEST (SwitchPRangeFunctor, SwitchAboveOne)
{
    TestSwitchPseudorange a;
    double s = 4.0;
    Vector3d res;
    (*a.f)(a.x.data(), a.v.data(), a.clk_bias.data(), a.x_e2n.data(), &s, res.data());

    EXPECT_MAT_NEAR(a.true_res, res, 1e-4);
}

TEST (SwitchPRangeFunctor, SwitchBelowZero)
{
    TestSwitchPseudorange a;
    double s = -4.0;
    Vector3d res;
    (*a.f)(a.x.data(), a.v.data(), a.clk_bias.data(), a.x_e2n.data(), &s, res.data());

    EXPECT_MAT_NEAR(Vector3d(0, 0, 1.0), res, 1e-4);
}


