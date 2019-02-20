#include "utils/trajectory_sim.h"

TrajectorySim::TrajectorySim(const string& yaml_file) :
    sim(false, 2)
{
    init(yaml_file);
    init_residual_blocks();
    init_estimator();
}

void TrajectorySim::init(const std::string& yaml_file)
{
    sim.load(yaml_file);

    switch_weight = 10.0;
    switch_dyn_weight = 2.0;

    x_e2n_hat = sim.X_e2n_;
    b << sim.accel_bias_, sim.gyro_bias_;
    bhat.setZero();

    measurements.resize(sim.satellites_.size());
    gtimes.resize(sim.satellites_.size());
    cov.resize(sim.satellites_.size());
    n = 0;
    new_node = false;
    s.resize(sim.satellites_.size(), N);
    shat.resizeLike(s);
}

void TrajectorySim::fix_origin()
{
    problem.AddResidualBlock(new XformNodeFactorAD(new XformNodeFunctor(sim.state().X.arr(), Matrix6d::Identity()*1e-6)),
                             NULL, xhat.data());
}

void TrajectorySim::init_residual_blocks()
{
    vhat.setZero();
    dt_hat.setZero();
    shat.setConstant(1.0);
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



void TrajectorySim::imuCallback(const double& t, const Vector6d& z, const Matrix6d& R)
{
    factor->integrate(t, z, R);
}

void TrajectorySim::rawGnssCallback(const GTime& t, const VecVec3& z, const VecMat3& R, std::vector<Satellite>& sats, const std::vector<bool>& slip)
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
}

void TrajectorySim::init_estimator()
{
    xhat.col(0) = sim.state().X.arr();
    x.col(0) = sim.state().X.arr();
    factors.push_back(new Imu3DFunctor(0, bhat));
    factor = factors[0];
    sim.register_estimator(this);
    t.push_back(sim.t_);
}

void TrajectorySim::addImuFactor()
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

void TrajectorySim::initNodePostionFromPointPos()
{
    Vector3d xn_ecef = sim.X_e2n_.t();
    WSG84::pointPositioning(gtimes[0], measurements, sim.satellites_, xn_ecef);
    xhat.template block<3,1>(0, n) = WSG84::ecef2ned(sim.X_e2n_, xn_ecef);
}

void TrajectorySim::addPseudorangeFactors()
{
    for (int i = 0; i < measurements.size(); i++)
    {
        problem.AddResidualBlock(new PRangeFactorAD(new PRangeFunctor(gtimes[i],
                                                                      measurements[i].topRows<2>(),
                                                                      sim.satellites_[i],
                                                                      sim.get_position_ecef(),
                                                                      cov[i].topLeftCorner<2,2>())),
                                 NULL,
                                 xhat.data() + (n)*7,
                                 vhat.data() + (n)*3,
                                 dt_hat.data(),
                                 x_e2n_hat.data());
        s(i, n) = (double)(sim.multipath_offset_[i] == 0);
    }
}

void TrajectorySim::addPseudorangeFactorsWithClockDynamics()
{
    for (int i = 0; i < measurements.size(); i++)
    {
        problem.AddResidualBlock(new PRangeFactorAD(new PRangeFunctor(gtimes[i],
                                                                      measurements[i].topRows<2>(),
                                                                      sim.satellites_[i],
                                                                      sim.get_position_ecef(),
                                                                      cov[i].topLeftCorner<2,2>())),
                                 NULL,
                                 xhat.data() + (n)*7,
                                 vhat.data() + (n)*3,
                                 dt_hat.data() + (n)*2,
                                 x_e2n_hat.data());
        s(i, n) = (double)(sim.multipath_offset_[i] == 0);
    }
    if (n < 1)
        return;

    double deltat = sim.t_ - t.back();
    dt_hat(0, n) = dt_hat(0, n-1) + deltat * dt_hat(1,n-1);
    dt_hat(1, n) = dt_hat(1, n-1);

    problem.AddResidualBlock(new ClockBiasFactorAD(new ClockBiasFunctor(deltat, dt_cov)),
                             NULL, dt_hat.data() + (n-1)*2, dt_hat.data() + n*2);

}

void TrajectorySim::addSwitchingPseudorangeFactors()
{
    for (int i = 0; i < measurements.size(); i++)
    {
        problem.AddResidualBlock(new SwitchPRangeFactorAD(new SwitchPRangeFunctor(gtimes[i],
                                                                      measurements[i].topRows<2>(),
                                                                      sim.satellites_[i],
                                                                      sim.get_position_ecef(),
                                                                      cov[i].topLeftCorner<2,2>(),
                                                                      shat(i,n), switch_weight)),
                                 NULL,
                                 xhat.data() + (n)*7,
                                 vhat.data() + (n)*3,
                                 dt_hat.data() + (n)*2,
                                 x_e2n_hat.data(),
                                 shat.data() + s.rows()*n+i);
        s(i, n) = (double)(sim.multipath_offset_[i] == 0);
        if (n > 1)
            problem.AddResidualBlock(new SwitchDynamicsFactorAD(new SwitchDynamicsFunctor(switch_dyn_weight)),
                                     NULL, shat.data() + (n-1)*s.rows() + i, shat.data() + n*s.rows() + i);
    }
    if (n < 1)
        return;

    double deltat = sim.t_ - t.back();
    dt_hat(0, n) = dt_hat(0, n-1) + deltat * dt_hat(1,n-1);
    dt_hat(1, n) = dt_hat(1, n-1);

    problem.AddResidualBlock(new ClockBiasFactorAD(new ClockBiasFunctor(deltat, dt_cov)),
                             NULL, dt_hat.data() + (n-1)*2, dt_hat.data() + n*2);
}

void TrajectorySim::addCarrierPhaseFactors()
{
    if (n < 1)
    {
        t0 = gtimes[0];
        Phi0.resize(measurements.size());
        for (int i = 0; i < measurements.size(); i++)
        {
            Phi0[i] = measurements[i](2);
        }
        return;
    }

    for (int i = 0; i < measurements.size(); i++)
    {
        problem.AddResidualBlock(new CarrierPhaseFactorAD(
                                     new CarrierPhaseFunctor(t0,
                                                             gtimes[i],
                                                             measurements[i](2) - Phi0[i],
                                                             sim.satellites_[i],
                                                             sim.get_position_ecef(),
                                                             cov[i](2,2))),
                                                             NULL,
                                                             xhat.data(),
                                                             xhat.data()+n*7,
                                                             dt_hat.data(),
                                                             dt_hat.data() + n*2,
                                                             x_e2n_hat.data());
    }
}

void TrajectorySim::addNodeCB(std::function<void(void)> cb)
{
    new_node_funcs.push_back(cb);
}

void TrajectorySim::run()
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

void TrajectorySim::solve()
{
    xhat0 = xhat;
    vhat0 = vhat;
    dt_hat0 = dt_hat;
    shat0 = shat;
    error0 = (xhat - x).array().abs().sum();


    ceres::Solver::Options options;
    options.max_num_iterations = 100;
    options.num_threads = 6;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;

    ceres::Solve(options, &problem, &summary);
}

void TrajectorySim::log(string filename)
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
                       dt.col(i),
                       s.col(i),
                       shat.col(i),
                       shat0.col(i));
    }
}

double TrajectorySim::TrajectorySim::final_error()
{
    return (xhat - x).array().abs().sum();
}
