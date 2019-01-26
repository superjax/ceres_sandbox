#include <fstream>

#include <ceres/ceres.h>
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <gtest/gtest.h>

#include "geometry/xform.h"
#include "multirotor_sim/simulator.h"
#include "multirotor_sim/controller.h"
#include "multirotor_sim/satellite.h"
#include "utils/estimator_wrapper.h"
#include "test_common.h"

#include "factors/pseudorange.h"


using namespace multirotor_sim;
using namespace ceres;
using namespace Eigen;
using namespace xform;


TEST (Pseudorange, TestCompile)
{
    GTime t;
    Vector2d rho;
    Satellite sat(1);
    Vector3d rec_pos;
    Matrix2d cov;
    PseudorangeCostFunction prange_factor(t, rho, sat, rec_pos, cov);
}

class TestPseudorange : public ::testing::Test
{
protected:
  TestPseudorange() :
    sat(1)
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

TEST_F (TestPseudorange, CheckResidualAtInit)
{
    Vector3d provo_lla{40.246184 * DEG2RAD , -111.647769 * DEG2RAD, 1387.997511};
    Vector3d rec_pos = WSG84::lla2ecef(provo_lla);
    Xformd x_e2n = WSG84::x_ecef2ned(rec_pos);

    Vector3d z;
    Vector2d rho;
    sat.computeMeasurement(time, rec_pos, Vector3d::Zero(), z);
    rho = z.topRows<2>();
    Matrix2d cov = (Vector2d{3.0, 0.4}).asDiagonal();

    PseudorangeCostFunction prange_factor(time, rho, sat, rec_pos, cov);

    Xformd x = Xformd::Identity();
    Vector3d v = Vector3d::Zero();
    Vector2d clk = Vector2d::Zero();
    Vector2d res = Vector2d::Zero();

    prange_factor(x.data(), v.data(), clk.data(), x_e2n.data(), res.data());

    EXPECT_MAT_NEAR(res, Vector2d::Zero(), 1e-4);
}

TEST_F (TestPseudorange, CheckResidualAfterMoving)
{
    Vector3d provo_lla{40.246184 * DEG2RAD , -111.647769 * DEG2RAD, 1387.997511};
    Vector3d rec_pos = WSG84::lla2ecef(provo_lla);
    Xformd x_e2n = WSG84::x_ecef2ned(rec_pos);

    Vector3d z;
    Vector2d rho;
    sat.computeMeasurement(time, rec_pos, Vector3d::Zero(), z);
    rho = z.topRows<2>();
    Matrix2d cov = (Vector2d{3.0, 0.4}).asDiagonal();

    PseudorangeCostFunction prange_factor(time, rho, sat, rec_pos, cov);

    Xformd x = Xformd::Identity();
    x.t() << 10, 0, 0;
    Vector3d p_ecef = WSG84::ned2ecef(x_e2n, x.t());
    Vector3d znew;
    sat.computeMeasurement(time, p_ecef, Vector3d::Zero(), znew);
    Vector2d true_res = Vector2d{std::sqrt(1/3.0), std::sqrt(1/0.4)}.asDiagonal() * (z - znew).topRows<2>();

    Vector3d v = Vector3d::Zero();
    Vector2d clk = Vector2d::Zero();
    Vector2d res = Vector2d::Zero();

    prange_factor(x.data(), v.data(), clk.data(), x_e2n.data(), res.data());

    EXPECT_MAT_NEAR(true_res, res, 1e-4);
}

