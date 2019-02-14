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

#include "factors/SE3.h"
#include "factors/imu_3d.h"
#include "factors/switch.h"
#include "factors/pseudorange.h"
#include "factors/clock_bias_dynamics.h"


using namespace multirotor_sim;
using namespace Eigen;
using namespace xform;

template<int N>
class TestSwitchPRangeTraj : public EstimatorBase
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  TestSwitchPRangeTraj() :
      sim(false, 2)
  {
      init();
      init_residual_blocks();
      init_estimator();
  }

  int n = 0;

  Simulator sim;
  Eigen::Matrix<double, 7, N> xhat0, xhat, x;
  Eigen::Matrix<double, 3, N> vhat0, vhat, v;
  Eigen::Matrix<double, 2, N> dt_hat0, dt_hat, dt;
  std::vector<double> t;
  Xformd x_e2n_hat;
  Vector6d b, bhat;
  Matrix2d dt_cov = Vector2d{1e-5, 1e-6}.asDiagonal();

  std::vector<Vector3d, aligned_allocator<Vector3d>> measurements;
  std::vector<Matrix2d, aligned_allocator<Matrix2d>> cov;
  std::vector<GTime> gtimes;

  ceres::Problem problem;
  bool new_node;
  std::vector<Imu3DFunctor*> factors;
  Imu3DFunctor* factor;
  std::vector<std::function<void(void)>> new_node_funcs;

  double error0;

  void init()
  {
      sim.load(raw_gps_yaml_file());

      x_e2n_hat = sim.X_e2n_;
      b << sim.accel_bias_, sim.gyro_bias_;
      bhat.setZero();

      measurements.resize(sim.satellites_.size());
      gtimes.resize(sim.satellites_.size());
      cov.resize(sim.satellites_.size());
      n = 0;
      new_node = false;
  }

  void init_residual_blocks()
  {
      vhat.setZero();
      dt_hat.setZero();
      for (int i = 0; i < N; i++)
      {
          xhat.col(i) = Xformd::Identity().elements();
          problem.AddParameterBlock(xhat.data() + i*7, 7, new XformParamAD());
          problem.AddParameterBlock(vhat.data() + i*3, 3);
          problem.AddParameterBlock(dt_hat.data()+i*2, 2);
      }
      problem.AddParameterBlock(x_e2n_hat.data(), 7, new XformParamAD());
      problem.SetParameterBlockConstant(x_e2n_hat.data());

  }



  void imuCallback(const double& t, const Vector6d& z, const Matrix6d& R) override
  {
      factor->integrate(t, z, R);
  }

  void rawGnssCallback(const GTime& t, const VecVec3& z, const VecMat3& R, std::vector<Satellite>& sats) override
  {
      int i = 0;
      for (auto sat : sats)
      {
          measurements[sat.idx_] = z[i];
          cov[sat.idx_] = R[i].topLeftCorner<2,2>();
          gtimes[sat.idx_] = t;
          i++;
          new_node = true;
      }
  }

  void init_estimator()
  {
      xhat.col(0) = sim.state().X.arr();
      x.col(0) = sim.state().X.arr();
      factors.push_back(new Imu3DFunctor(0, bhat));
      factor = factors[0];
      sim.register_estimator(this);
      t.push_back(sim.t_);
  }

  void addImuFactor()
  {
      if (n < 1)
          return;
      factor->estimateXj(xhat.data()+7*(n-1), vhat.data()+3*(n-1), xhat.data()+7*(n), vhat.data()+3*(n));
      factor->finished();
      problem.AddResidualBlock(new Imu3DFactorAD(factor), NULL,
                               xhat.data()+7*(n-1), xhat.data()+7*(n),
                               vhat.data()+3*(n-1), vhat.data()+3*(n),
                               bhat.data());
      factors.push_back(new Imu3DFunctor(sim.t_, bhat));
      factor = factors.back();
  }

  void initNodePostionFromPointPos()
  {
      Vector3d xn_ecef = sim.X_e2n_.t();
      WSG84::pointPositioning(gtimes[0], measurements, sim.satellites_, xn_ecef);
      xhat.template block<3,1>(0, n) = WSG84::ecef2ned(sim.X_e2n_, xn_ecef);
  }

  void addPseudorangeFactors()
  {
      for (int i = 0; i < measurements.size(); i++)
      {
          problem.AddResidualBlock(new PRangeFactorAD(new PRangeFunctor(gtimes[i],
                                                                        measurements[i].topRows<2>(),
                                                                        sim.satellites_[i],
                                                                        sim.get_position_ecef(),
                                                                        cov[i])),
                                   NULL,
                                   xhat.data() + (n)*7,
                                   vhat.data() + (n)*3,
                                   dt_hat.data(),
                                   x_e2n_hat.data());
      }
  }

  void addPseudorangeFactorsWithClockDynamics()
  {
      for (int i = 0; i < measurements.size(); i++)
      {
          problem.AddResidualBlock(new PRangeFactorAD(new PRangeFunctor(gtimes[i],
                                                                        measurements[i].topRows<2>(),
                                                                        sim.satellites_[i],
                                                                        sim.get_position_ecef(),
                                                                        cov[i])),
                                   NULL,
                                   xhat.data() + (n)*7,
                                   vhat.data() + (n)*3,
                                   dt_hat.data() + (n)*2,
                                   x_e2n_hat.data());
      }
      if (n < 1)
          return;

      double deltat = sim.t_ - t.back();
      dt_hat(0, n) = dt_hat(0, n-1) + deltat * dt_hat(1,n-1);
      dt_hat(1, n) = dt_hat(1, n-1);

      problem.AddResidualBlock(new ClockBiasFactorAD(new ClockBiasFunctor(deltat, dt_cov)),
                               NULL, dt_hat.data() + (n-1)*2, dt_hat.data() + n*2);

  }

  void addNodeCB(std::function<void(void)> cb)
  {
      new_node_funcs.push_back(cb);
  }

  void run()
  {
      //    problem.AddResidualBlock(new XformNodeFactorAD(new XformNodeFunctor(sim.state().X.arr(), Matrix6d::Identity() * 1e-8)),
      //                             NULL, xhat.data());
      while (n < N)
      {
          sim.run();

          if (new_node)
          {
              new_node = false;
              for (auto fun : new_node_funcs)
              {
                  fun();
              }

              x.col(n) = sim.state().X.elements();
              v.col(n) = sim.state().v;
              dt.col(n) << sim.clock_bias_, sim.clock_bias_rate_    ;
              n++;
              t.push_back(sim.t_);
          }
      }
  }

  void solve()
  {
      xhat0 = xhat;
      vhat0 = vhat;
      dt_hat0 = dt_hat;
      error0 = (xhat - x).array().abs().sum();


      ceres::Solver::Options options;
      options.max_num_iterations = 100;
      options.num_threads = 6;
      options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
      options.minimizer_progress_to_stdout = false;
      ceres::Solver::Summary summary;

      ceres::Solve(options, &problem, &summary);
  }

  void log(string filename)
  {
      Logger<double> log(filename);

      for (int i = 0; i < N; i++)
      {
          log.log(t[i]);
          log.logVectors(xhat0.col(i),
                         vhat0.col(i),
                         xhat.col(i),
                         vhat.col(i),
                         x.col(i),
                         v.col(i),
                         dt_hat0.col(i),
                         dt_hat.col(i),
                         dt.col(i));
      }
  }

  double final_error()
  {
      return (xhat - x).array().abs().sum();
  }

};

TEST (SwitchPseudorange, ImuTrajectory)
{
    TestSwitchPRangeTraj<100> a;

    a.addNodeCB([&a](){a.addImuFactor();});
    a.addNodeCB([&a](){a.initNodePostionFromPointPos();});
    a.addNodeCB([&a](){a.addPseudorangeFactorsWithClockDynamics();});

    a.run();
    a.solve();
    a.log("/tmp/ceres_sandbox/SwitchPseudorange.ImuTrajectory.log");

    ASSERT_LE(a.final_error(), a.error0);
}



