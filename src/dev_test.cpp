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

#include "factors/pseudorange.h"
#include "factors/SE3.h"
#include "factors/imu_3d.h"
#include "factors/clock_bias_dynamics.h"
#include "factors/carrier_phase.h"


using namespace multirotor_sim;
using namespace Eigen;
using namespace xform;


TEST(CarrierPhase, ImuTrajectory)
{
    Simulator sim(false, 2);
    sim.load(raw_gps_yaml_file());

    const int N = 100;
    int n = 0;

    Eigen::Matrix<double, 7, N> xhat, x;
    Eigen::Matrix<double, 3, N> vhat, v;
    Eigen::Matrix<double, 2, N> tauhat, tau;
    std::vector<double> t;
    Vector6d b, bhat;
    b << sim.accel_bias_, sim.gyro_bias_;
    bhat.setZero();
    Xformd x_e2n_hat = sim.X_e2n_;

    std::default_random_engine rng;
    std::normal_distribution<double> normal;

    std::vector<Vector3d, aligned_allocator<Vector3d>> measurements;
    std::vector<Matrix3d, aligned_allocator<Matrix3d>> cov;
    std::vector<GTime> gtimes;
    measurements.resize(sim.satellites_.size());
    gtimes.resize(sim.satellites_.size());
    cov.resize(sim.satellites_.size());

    Matrix2d tau_cov = Vector2d{1e-5, 1e-6}.asDiagonal();

    ceres::Problem problem;

    for (int i = 0; i < N; i++)
    {
        xhat.col(i) = Xformd::Identity().elements();
        problem.AddParameterBlock(xhat.data() + i*7, 7, new XformParamAD());
        vhat.setZero();
        problem.AddParameterBlock(vhat.data() + i*3, 3);
        tauhat.setZero();
        problem.AddParameterBlock(tauhat.data() + i*2, 2);
    }
    problem.AddParameterBlock(x_e2n_hat.data(), 7, new XformParamAD());
    problem.SetParameterBlockConstant(x_e2n_hat.data());

    xhat.col(0) = sim.state().X.arr();

    std::vector<Imu3DFunctor*> factors;
    factors.push_back(new Imu3DFunctor(0, bhat));

    Imu3DFunctor* factor = factors[0];
    auto imu_cb = [&factor](const double& t, const Vector6d& z, const Matrix6d& R)
    {
        factor->integrate(t, z, R);
    };

    bool new_node = false;
    auto raw_gnss_cb = [&measurements, &new_node, &cov, &gtimes]
            (const GTime& t, const VecVec3& z, const VecMat3& R, std::vector<Satellite>& sats)
    {
        int i = 0;
        for (auto sat : sats)
        {
            measurements[sat.idx_] = z[i];
            cov[sat.idx_] = R[i];
            gtimes[sat.idx_] = t;
            i++;
            new_node = true;
        }
    };

    EstimatorWrapper est;
    est.register_raw_gnss_cb(raw_gnss_cb);
    est.register_imu_cb(imu_cb);
    sim.register_estimator(&est);

    t.push_back(sim.t_);
    GTime t0;
    std::vector<double> Phi0(sim.satellites_.size());
    while (n < N-1)
    {
        sim.run();

        if (new_node)
        {
            new_node = false;
            factor->estimateXj(xhat.data()+7*(n), vhat.data()+3*(n), xhat.data()+7*(n+1), vhat.data()+3*(n+1));
            factor->finished();
            problem.AddResidualBlock(new Imu3DFactorAD(factor), NULL,
                                     xhat.data()+7*(n), xhat.data()+7*(n+1),
                                     vhat.data()+3*(n), vhat.data()+3*(n+1),
                                     bhat.data());
            factors.push_back(new Imu3DFunctor(sim.t_, bhat));
            factor = factors.back();

            Vector3d xn_ecef = sim.X_e2n_.t();
            WSG84::pointPositioning(gtimes[0], measurements, sim.satellites_, xn_ecef);
            xhat.block<3,1>(0, n+1) = WSG84::ecef2ned(sim.X_e2n_, xn_ecef);


            for (int i = 0; i < measurements.size(); i++)
            {
                problem.AddResidualBlock(new PRangeFactorAD(
                                             new PRangeFunctor(gtimes[i],
                                                               measurements[i].topRows<2>(),
                                                               sim.satellites_[i],
                                                               sim.get_position_ecef(),
                                                               cov[i].topLeftCorner<2,2>())),
                                         NULL,
                                         xhat.data() + n*7,
                                         vhat.data() + n*3,
                                         tauhat.data() + n*2,
                                         x_e2n_hat.data());
                if (n > 0)
                {
                    problem.AddResidualBlock(new ClockBiasFactorAD(new ClockBiasFunctor(sim.t_ - t.back(), tau_cov)),
                                             NULL, tauhat.data() + (n-1)*2, tauhat.data() + n*2);
//                    problem.AddResidualBlock(new CarrierPhaseFactorAD(
//                                                 new CarrierPhaseFunctor(t0,
//                                                                         gtimes[i],
//                                                                         measurements[i](2) - Phi0[i],
//                                                                         sim.satellites_[i],
//                                                                         sim.get_position_ecef(),
//                                                                         cov[i](2,2))),
//                                                                         NULL,
//                                                                         xhat.data(),
//                                                                         xhat.data()+n*7,
//                                                                         tauhat.data(),
//                                                                         tauhat.data() + n*2,
//                                                                         x_e2n_hat.data());
                }
                else
                {
                    t0 = gtimes[i];
                    Phi0[i] = measurements[i](2);
                }
            }
            x.col(n) = sim.dyn_.get_state().X.elements();
            v.col(n) = sim.dyn_.get_state().q.rota(sim.dyn_.get_state().v);
            tau.col(n) << sim.clock_bias_, sim.clock_bias_rate_;
            t.push_back(sim.t_);
            n++;

        }
    }


    ceres::Solver::Options options;
    options.max_num_iterations = 100;
    options.num_threads = 6;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;

    MatrixXd xhat0 = xhat;
    MatrixXd vhat0 = vhat;
    MatrixXd tauhat0 = tauhat;
    double error0 = (xhat - x).array().abs().sum();

    ceres::Solve(options, &problem, &summary);

    Logger<double> log("/tmp/ceres_sandbox/CarrierPhase.ImuTrajectory.log");

    for (int i = 0; i < N; i++)
    {
        log.log(t[i]);
        log.logVectors(xhat0.col(i),
                       vhat0.col(i),
                       xhat.col(i),
                       vhat.col(i),
                       x.col(i),
                       v.col(i),
                       tauhat0.col(i),
                       tauhat.col(i),
                       tau.col(i));
    }
    double error = (xhat - x).array().abs().sum();
    ASSERT_LE(error, error0);

}

