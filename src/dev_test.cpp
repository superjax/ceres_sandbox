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


using namespace multirotor_sim;
using namespace Eigen;
using namespace xform;


TEST(Pseudorange, Trajectory)
{
    ReferenceController cont;
    cont.load("../params/sim_params.yaml");
    Simulator sim(cont, cont, false, 2);
    sim.load("../params/sim_params.yaml");

    const int N = 100;
    int n = 0;

    Matrix<double, 7, N> xhat, x;
    Matrix<double, 3, N> vhat, v;
    std::vector<double> t;
    Xformd x_e2n_hat = sim.x_e2n_;
    Vector2d clk_bias_hat, clk_bias;

    std::default_random_engine rng;
    std::normal_distribution<double> normal;

    std::vector<Vector2d, aligned_allocator<Vector2d>> measurements;
    std::vector<Matrix2d, aligned_allocator<Matrix2d>> cov;
    std::vector<GTime> gtimes;
    measurements.resize(sim.satellites_.size());
    gtimes.resize(sim.satellites_.size());
    cov.resize(sim.satellites_.size());

    ceres::Problem problem;

    for (int i = 0; i < N; i++)
    {
        xhat.col(i) = Xformd::Identity().elements();
        problem.AddParameterBlock(xhat.data() + i*7, 7, new XformParamAD());
        vhat.setZero();
        problem.AddParameterBlock(vhat.data() + i*3, 3);
    }
    clk_bias_hat.setZero();
    problem.AddParameterBlock(clk_bias_hat.data(), 2);
    problem.AddParameterBlock(x_e2n_hat.data(), 7, new XformParamAD());
    problem.SetParameterBlockConstant(x_e2n_hat.data());

    bool new_node = false;
    auto raw_gnss_cb = [&measurements, &new_node, &cov, &gtimes]
            (const GTime& t, const Vector3d& z, const Matrix3d& R, Satellite& sat)
    {
        measurements[sat.idx_] = z.topRows<2>();
        cov[sat.idx_] = R.topLeftCorner<2,2>();
        gtimes[sat.idx_]=t;
        new_node = true;
    };

    EstimatorWrapper est;
    est.register_raw_gnss_cb(raw_gnss_cb);
    sim.register_estimator(&est);

    while (n < N)
    {
        sim.run();

        if (new_node)
        {
            new_node = false;
            for (int i = 0; i < measurements.size(); i++)
            {
                problem.AddResidualBlock(new PRangeFactorAD(new PRangeFunctor(gtimes[i],
                                                                                  measurements[i],
                                                                                  sim.satellites_[i],
                                                                                  sim.get_position_ecef(),
                                                                                  cov[i])),
                                                      NULL,
                                                      xhat.data() + n*7,
                                                      vhat.data() + n*3,
                                                      clk_bias_hat.data(),
                                                      x_e2n_hat.data());
            }
            x.col(n) = sim.dyn_.get_state().X.elements();
            v.col(n) = sim.dyn_.get_state().q.rota(sim.dyn_.get_state().v);
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
    Xformd x_e2n_hat0 = x_e2n_hat;
    Vector2d clk_bias_hat0 = clk_bias_hat;

    ceres::Solve(options, &problem, &summary);

    Logger<double> log("/tmp/ceres_sandbox/Pseudorange.Trajectory.log");

    for (int i = 0; i < N; i++)
    {
        log.log(t[i]);
        log.logVectors(xhat0.col(i),
                       vhat0.col(i),
                       xhat.col(i),
                       vhat.col(i),
                       x.col(i),
                       v.col(i));
    }

}

