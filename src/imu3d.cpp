#include <fstream>

#include <ceres/ceres.h>
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <gtest/gtest.h>

#include "factors/range_1d.h"
#include "factors/SE3.h"
#include "factors/imu_3d.h"

#include "geometry/xform.h"
#include "geometry/support.h"
#include "multirotor_sim/simulator.h"
#include "multirotor_sim/controller.h"
#include "utils/jac.h"
#include "utils/estimator_wrapper.h"
#include "utils/logger.h"
#include "test_common.h"

using namespace ceres;
using namespace Eigen;
using namespace std;
using namespace xform;
using namespace multirotor_sim;


TEST(Imu3D, compile)
{
    Imu3DFunctor imu;
}

TEST(Imu3D, reset)
{
    Imu3DFunctor imu;
    Vector6d b0;
    b0 << 0, 1, 2, 3, 4, 5;
    imu.reset(0, b0);
}

TEST(Imu3D, Propagation)
{
    Simulator multirotor;
    multirotor.load(imu_only_sim());

    typedef Imu3DFunctor IMU;
    IMU imu;
    Vector6d b0;
    b0.setZero();
    Matrix6d cov = Matrix6d::Identity() * 1e-3;

    multirotor.run();
    imu.reset(multirotor.t_, b0);
    Xformd x0 = multirotor.state().X;
    Vector3d v0 = multirotor.state().v;

    Logger<double> log("/tmp/ceres_sandbox/Imu3D.CheckPropagation.log");

    Xformd xhat = multirotor.state().X;
    Vector3d vhat = multirotor.state().v;
    log.log(multirotor.t_);
    log.logVectors(xhat.elements(), vhat, multirotor.state().X.elements(),
                   multirotor.state().v, multirotor.imu());

    double next_reset = 1.0;
    multirotor.tmax_ = 10.0;
    while (multirotor.run())
    {
        imu.integrate(multirotor.t_, multirotor.imu(), cov);

        if (std::abs(multirotor.t_ - next_reset) <= multirotor.dt_ /2.0)
        {
            imu.reset(multirotor.t_, b0);
            x0 = multirotor.state().X;
            v0 = multirotor.state().v;
            next_reset += 1.0;
        }

        imu.estimateXj(x0.data(), v0.data(), xhat.data(), vhat.data());
        log.log(multirotor.t_);
        log.logVectors(xhat.elements(), vhat, multirotor.state().X.elements(), multirotor.state().v, multirotor.imu());
        EXPECT_MAT_NEAR(xhat.t(), multirotor.state().p, 0.076);
        EXPECT_QUAT_NEAR(xhat.q(), multirotor.state().q, 0.0033);
        EXPECT_MAT_NEAR(vhat, multirotor.state().v, 0.04);
    }
}


TEST(Imu3D, ErrorStateDynamics)
{
    typedef Imu3DFunctor IMU;
    IMU y;
    IMU yhat;
    Vector9d dy;
    double t = 0;
    const double Tmax = 10.0;
    static const double dt = 0.001;

    Vector6d bias;
    bias.setZero();
    y.reset(t, bias);
    yhat.reset(t, bias);
    IMU::boxplus(y.y_, Vector9d::Constant(0.01), yhat.y_);
    IMU::boxminus(y.y_, yhat.y_, dy);

    Vector10d y_check;
    IMU::boxplus(yhat.y_, dy, y_check);
    ASSERT_MAT_NEAR(y.y_, y_check, 1e-8);

    Vector6d u, eta;
    u.setZero();
    eta.setZero();

    std::default_random_engine gen;
    std::normal_distribution<double> normal;

    Logger<double> log("/tmp/ceres_sandbox/Imu3d.CheckDynamics.log");


    Matrix6d cov = Matrix6d::Identity() * 1e-3;
    Vector9d dydot;
    log.log(t);
    log.logVectors(y.y_, yhat.y_, dy, y_check, u);
    for (int i = 0; i < Tmax/dt; i++)
    {
        u += dt * randomNormal<double,6,1>(normal, gen);
        t += dt;
        y.errorStateDynamics(y.y_, dy, u, eta, dydot);
        dy += dydot * dt;

        y.integrate(t, u, cov);
        yhat.integrate(t, u, cov);
        IMU::boxplus(yhat.y_, dy, y_check);
        log.log(t);
        log.logVectors(y.y_, yhat.y_, dy, y_check, u);
        ASSERT_MAT_NEAR(y.y_, y_check, t > 0.3 ? 5e-6*t*t : 2e-7);
    }
}


TEST(Imu3D, DynamicsJacobians)
{
    Matrix6d cov = Matrix6d::Identity()*1e-3;

    Vector6d b0;
    Vector10d y0;
    Vector6d u0;
    Vector6d eta0;
    Vector9d ydot;
    Vector9d dy0;

    Matrix9d A;
    Eigen::Matrix<double, 9, 6> B;

    for (int i = 0; i < 100; i++)
    {
        b0.setRandom();
        y0.setRandom();
        y0.segment<4>(6) = Quatd::Random().elements();
        u0.setRandom();

        eta0.setZero();
        dy0.setZero();

        Imu3DFunctor f;
        f.reset(0, b0);
        f.dynamics(y0, u0, ydot, A, B);
        Vector9d dy0;

        auto yfun = [&y0, &cov, &b0, &u0, &eta0](const Vector9d& dy)
        {
            Imu3DFunctor f;
            f.reset(0, b0);
            Vector9d dydot;
            f.errorStateDynamics(y0, dy, u0, eta0, dydot);
            return dydot;
        };
        auto etafun = [&y0, &cov, &b0, &dy0, &u0](const Vector6d& eta)
        {
            Imu3DFunctor f;
            f.reset(0, b0);
            Vector9d dydot;
            f.errorStateDynamics(y0, dy0, u0, eta, dydot);
            return dydot;
        };

        Matrix9d AFD = calc_jac(yfun, dy0);
        Eigen::Matrix<double, 9, 6> BFD = calc_jac(etafun, u0);

//        cout << "A\n" << A << "\nAFD\n" << AFD << "\n\n";
//        cout << "B\n" << B << "\nBFD\n" << BFD << "\n\n";

        ASSERT_MAT_NEAR(AFD, A, 1e-7);
        ASSERT_MAT_NEAR(BFD, B, 1e-7);
    }
}

TEST(Imu3D, BiasJacobians)
{
    YAML::Node node;
    node = YAML::LoadFile("../lib/multirotor_sim/params/sim_params.yaml");
    ofstream tmp("/tmp/ceres_sandbox.tmp.yaml");
    node["imu_enabled"] =  true;
    node["alt_enabled"] =  false;
    node["mocap_enabled"] =  false;
    node["vo_enabled"] =  false;
    node["camera_enabled"] =  false;
    node["gnss_enabled"] =  false;
    node["raw_gnss_enabled"] =  false;
    tmp << node;
    tmp.close();
    Simulator multirotor(false, 1);
    multirotor.load("/tmp/ceres_sandbox.tmp.yaml");
    std::vector<Vector6d,Eigen::aligned_allocator<Vector6d>> meas;
    std::vector<double> t;
    multirotor.dt_ = 0.001;

    while (multirotor.t_ < 0.1)
    {
        multirotor.run();
        meas.push_back(multirotor.imu());
        t.push_back(multirotor.t_);
    }

    Matrix6d cov = Matrix6d::Identity()*1e-3;
    Vector6d b0;
    Eigen::Matrix<double, 9, 6> J, JFD;

    b0.setZero();
    Imu3DFunctor f;
    f.reset(0, b0);
    Vector10d y0 = f.y_;
    for (int i = 0; i < meas.size(); i++)
    {
        f.integrate(t[i], meas[i], cov);
    }
    J = f.J_;

    auto fun = [&cov, &meas, &t, &y0](const Vector6d& b0)
    {
        Imu3DFunctor f;
        f.reset(0, b0);
        for (int i = 0; i < meas.size(); i++)
        {
            f.integrate(t[i], meas[i], cov);
        }
        return f.y_;
    };
    auto bm = [](const MatrixXd& x1, const MatrixXd& x2)
    {
        Vector9d dx;
        Imu3DFunctor::boxminus(x1, x2, dx);
        return dx;
    };

    JFD = calc_jac(fun, b0, nullptr, nullptr, bm, 1e-5);
//    std::cout << "FD:\n" << JFD << "\nA:\n" << J <<std::endl;
    ASSERT_MAT_NEAR(J, JFD, 1e-4);
}

TEST(Imu3D, MultiWindow)
{
    Simulator multirotor(false, 2);
    multirotor.load(imu_only_sim());

    const int N = 10;

    Vector6d b, bhat;
    b.block<3,1>(0,0) = multirotor.accel_bias_;
    b.block<3,1>(3,0) = multirotor.gyro_bias_;
    bhat.setZero();

    Problem problem;

    Eigen::MatrixXd xhat, x;
    Eigen::MatrixXd vhat, v;
    xhat.resize(7, N+1);
    x.resize(7, N+1);
    vhat.resize(3, N+1);
    v.resize(3, N+1);

    std::normal_distribution<double> normal;
    std::default_random_engine rng;

    xhat.col(0) = multirotor.state().X.arr_;
    vhat.col(0) = multirotor.dyn_.get_state().v;
    x.col(0) = multirotor.state().X.arr_;
    v.col(0) = multirotor.dyn_.get_state().v;

    xhat.setZero();
    vhat.setZero();
    xhat.row(3).setConstant(1.0);

    for (int n = 0; n < N; n++)
    {
        problem.AddParameterBlock(xhat.data()+7*n, 7, new XformParamAD());
        problem.AddParameterBlock(vhat.data()+3*n, 3);
    }
    problem.AddParameterBlock(bhat.data(), 6);


    std::vector<Imu3DFunctor*> factors;
    factors.push_back(new Imu3DFunctor(0, bhat));

    int node = 0;
    Imu3DFunctor* factor = factors[node];
    auto imu_cb = [&factor](const double& t, const Vector6d& z, const Matrix6d& R)
    {
        factor->integrate(t, z, R);
    };

    EstimatorWrapper est;
    est.register_imu_cb(imu_cb);
    multirotor.register_estimator(&est);


    // Integrate for N frames
    std::vector<double> t;
    t.push_back(multirotor.t_);
    double node_dt = 1.0;
    double next_node = node_dt;
    Matrix6d P = Matrix6d::Identity() * 0.1;
    problem.AddResidualBlock(new XformNodeFactorAD(new XformNodeFunctor(multirotor.state().X.arr_, P)), NULL, xhat.data());
    while (node < N)
    {
        multirotor.run();

        if (std::abs(multirotor.t_ - next_node) < node_dt / 2.0)
        {
            next_node += node_dt;
            t.push_back(multirotor.t_);
            node += 1;

            // estimate next node pose and velocity with IMU preintegration
            factor->estimateXj(xhat.data()+7*(node-1), vhat.data()+3*(node-1), xhat.data()+7*(node), vhat.data()+3*(node));
            // Calculate the Information Matrix of the IMU factor
            factor->finished();

            // Save off True Pose and Velocity for Comparison
            x.col(node) = multirotor.state().X.arr_;
            v.col(node) = multirotor.state().v;

            xhat.col(node) = (multirotor.state().X + randomNormal<double, 6, 1>(normal, rng)).arr();

            // Add IMU factor to graph
            problem.AddResidualBlock(new Imu3DFactorAD(factor), NULL,
                                     xhat.data()+7*(node-1), xhat.data()+7*node,
                                     vhat.data()+3*(node-1), vhat.data()+3*node,
                                     bhat.data());

            // Start a new Factor
            factors.push_back(new Imu3DFunctor(multirotor.t_, bhat));
            factor = factors[node];
            problem.AddResidualBlock(new XformNodeFactorAD(new XformNodeFunctor(multirotor.state().X.arr_, P)), NULL, xhat.data()+7*(node));
        }
    }

    Solver::Options options;
    options.max_num_iterations = 100;
    options.num_threads = 6;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = false;
    Solver::Summary summary;
    Logger<double> log("/tmp/ceres_sandbox/Imu3d.MultiWindow.log");

    MatrixXd xhat0 = xhat;
    MatrixXd vhat0 = vhat;

//    cout << "xhat0\n" << xhat.transpose() << endl;
//    cout << "bhat0\n" << bhat.transpose() << endl;

    ceres::Solve(options, &problem, &summary);
    double error = (b - bhat).norm();

//    cout << summary.FullReport();
//    cout << "x\n" << x.transpose() << endl;
//    cout << "xhatf\n" << xhat.transpose() << endl;
//    cout << "b\n" << b.transpose() << endl;
//    cout << "bhat\n" << bhat.transpose() << endl;
//    cout << "e " << error << endl;
    EXPECT_LE(error, 0.2);


    for (int i = 0; i <= N; i++)
    {
        log.log(t[i]);
        log.logVectors(xhat0.col(i), vhat0.col(i), xhat.col(i), vhat.col(i), x.col(i), v.col(i));
    }
}
