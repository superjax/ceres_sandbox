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
#include "simulator.h"
#include "utils/jac.h"

using namespace ceres;
using namespace Eigen;
using namespace std;
using namespace xform;

Vector10d boxplus(const Vector10d& y, const Vector9d& dy)
{
    Vector10d yp;
    yp.block<3,1>(0,0) = y.block<3,1>(0,0) + dy.block<3,1>(0,0);
    yp.block<3,1>(3,0) = y.block<3,1>(3,0) + dy.block<3,1>(3,0);
    yp.block<4,1>(6,0) = (Quatd(y.block<4,1>(6,0)) + dy.block<3,1>(6,0)).elements();
    return yp;
}

Vector9d boxminus(const Vector10d& y1, const Vector10d& y2)
{
    Vector9d out;
    out.block<3,1>(0,0) = y1.block<3,1>(0,0) - y2.block<3,1>(0,0);
    out.block<3,1>(3,0) = y1.block<3,1>(3,0) - y2.block<3,1>(3,0);
    out.block<3,1>(6,0) = Quatd(y1.block<4,1>(6,0)) - Quatd(y2.block<4,1>(6,0));
    return out;
}

TEST(Imu3D, CheckDynamicsJacobians)
{
    Matrix6d cov = Matrix6d::Identity()*1e-3;

    Vector6d b0;
    Vector10d y0;
    Vector6d u0;
    Vector9d ydot;

    Matrix9d A;
    Eigen::Matrix<double, 9, 6> B;
    Eigen::Matrix<double, 9, 6> C;

    for (int i = 0; i < 100; i++)
    {
        b0.setRandom();
        y0.setRandom();
        y0.segment<4>(6) = Quatd::Random().elements();
        u0.setRandom();
        Imu3DFactorCostFunction f(0, b0, cov);
        f.dynamics(y0, u0, ydot, A, B, C);

        auto yfun = [cov, b0, u0](Vector10d y)
        {
            Imu3DFactorCostFunction f(0, b0, cov);
            Vector9d ydot;
            Matrix9d A;
            Eigen::Matrix<double, 9, 6> B;
            Eigen::Matrix<double, 9, 6> C;
            f.dynamics(y, u0, ydot, A, B, C);
            return ydot;
        };
        auto bfun = [cov, y0, u0](Vector6d b)
        {
            Imu3DFactorCostFunction f(0, b, cov);
            Vector9d ydot;
            Matrix9d A;
            Eigen::Matrix<double, 9, 6> B;
            Eigen::Matrix<double, 9, 6> C;
            f.dynamics(y0, u0, ydot, A, B, C);
            return ydot;
        };
        auto ufun = [cov, b0, y0](Vector6d u)
        {
            Imu3DFactorCostFunction f(0, b0, cov);
            Vector9d ydot;
            Matrix9d A;
            Eigen::Matrix<double, 9, 6> B;
            Eigen::Matrix<double, 9, 6> C;
            f.dynamics(y0, u, ydot, A, B, C);
            return ydot;
        };

        Matrix9d AFD = calc_jac(yfun, y0, boxminus, boxplus);
        Eigen::Matrix<double, 9, 6> BFD = calc_jac(ufun, u0);
        Eigen::Matrix<double, 9, 6> CFD = calc_jac(bfun, b0);

//            cout << "AFD: \n" << AFD << "\n";
//            cout << "AA: \n" << A << "\n\n";

//            cout << "BFD: \n" << BFD << "\n";
//            cout << "BA: \n" << B << "\n\n";

//            cout << "CFD: \n" << CFD << "\n";
//            cout << "CA: \n" << C << "\n\n";
        EXPECT_LE((AFD - A).array().square().sum(), 1e-8);
        EXPECT_LE((BFD - B).array().square().sum(), 1e-8);
        EXPECT_LE((CFD - C).array().square().sum(), 1e-8);
    }
}

TEST(Imu3D, CheckBiasJacobians)
{
    Simulator multirotor(false);
    multirotor.load("../lib/multirotor_sim/params/sim_params.yaml");
    std::vector<Vector6d,Eigen::aligned_allocator<Vector6d>> meas;
    std::vector<double> t;

    // Integrate for 1 second
    while (multirotor.t_ < 1.0)
    {
        multirotor.run();
        meas.push_back(multirotor.get_imu_prev());
        t.push_back(multirotor.t_);
    }

    Matrix6d cov = Matrix6d::Identity()*1e-3;

    Vector6d b0;

    Eigen::Matrix<double, 9, 6> J, JFD;

    b0.setZero();
    Imu3DFactorCostFunction f(0, b0, cov);
    Vector10d y0 = f.y_;
    for (int i = 0; i < meas.size(); i++)
    {
        f.integrate(t[i], meas[i]);
    }
    J = f.J_;

    auto fun = [cov, meas, t, y0](Vector6d b0)
    {
        Imu3DFactorCostFunction f(0, b0, cov);
        for (int i = 0; i < meas.size(); i++)
        {
            f.integrate(t[i], meas[i]);
        }
        return f.y_;
    };
    JFD = calc_jac(fun, b0, nullptr, nullptr, boxminus, nullptr);

    cout << "J:\n" << J << "\n\n";
    cout << "JFD:\n" << JFD << "\n\n";
}

TEST(Imu3D, SingleWindow)
{
    Simulator multirotor(false);
    multirotor.load("../lib/multirotor_sim/params/sim_params.yaml");

    Vector6d b, bhat;
    b.block<3,1>(0,0) = multirotor.get_accel_bias();
    b.block<3,1>(3,0) = multirotor.get_gyro_bias();
    bhat.setZero();

    Problem problem;

    Eigen::Matrix<double, 7, 2> xhat, x;
    Eigen::Matrix<double, 3, 2> vhat, v;

    xhat.col(0) = multirotor.get_pose().arr_;
    vhat.col(0) = multirotor.dyn_.get_state().segment<3>(dynamics::VX);
    x.col(0) = multirotor.get_pose().arr_;
    v.col(0) = multirotor.dyn_.get_state().segment<3>(dynamics::VX);

    // Anchor origin node (pose and velocity)
    problem.AddParameterBlock(xhat.data(), 7, new XformAutoDiffParameterization());
    problem.SetParameterBlockConstant(xhat.data());
    problem.AddParameterBlock(vhat.data(), 3);
    problem.SetParameterBlockConstant(vhat.data());

    // Declare the bias parameters
    problem.AddParameterBlock(bhat.data(), 6);

    std::vector<Simulator::measurement_t, Eigen::aligned_allocator<Simulator::measurement_t>> meas_list;
    multirotor.get_measurements(meas_list);

    Imu3DFactorCostFunction *factor = new Imu3DFactorCostFunction(0, bhat, multirotor.get_imu_noise_covariance());

    // Integrate for 1 second
    while (multirotor.t_ < 1.0)
    {
        multirotor.run();
        factor->integrate(multirotor.t_, multirotor.get_imu_prev());
        multirotor.get_measurements(meas_list);
    }

    factor->estimate_xj(xhat.data(), vhat.data(), xhat.data()+7, vhat.data()+3);
    factor->finished();

    // Declare the new parameters
    problem.AddParameterBlock(xhat.data()+7, 7, new XformAutoDiffParameterization());
    problem.AddParameterBlock(vhat.data()+3, 3);

    // Add IMU factor to graph
    problem.AddResidualBlock(new Imu3DFactorAutoDiff(factor), NULL, xhat.data(), xhat.data()+7, vhat.data(), vhat.data()+3, bhat.data());

    // Add measurement of final pose
    x.col(1) = multirotor.get_pose().arr_;
    Matrix6d P = Matrix6d::Identity() * 0.1;
    problem.AddResidualBlock(new XformNodeFactorAutoDiff(new XformNodeFactorCostFunction(x.col(1), P)), NULL, xhat.data()+7);


    Solver::Options options;
    options.max_num_iterations = 25;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    Solver::Summary summary;

    cout << "xhat0\n" << xhat.transpose() << endl;
    cout << "bhat0\n" << bhat.transpose() << endl;

    ceres::Solve(options, &problem, &summary);
    summary.FullReport();

    cout << "x\n" << x.transpose() << endl;
    cout << "xhat0\n" << xhat.transpose() << endl;
    cout << "b\n" << b.transpose() << endl;
    cout << "bhat\n" << bhat.transpose() << endl;
}
