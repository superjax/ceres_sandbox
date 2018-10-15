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

        auto yfun = [&cov, &b0, &u0](const Vector10d& y)
        {
            Imu3DFactorCostFunction f(0, b0, cov);
            Vector9d ydot;
            Matrix9d A;
            Eigen::Matrix<double, 9, 6> B;
            Eigen::Matrix<double, 9, 6> C;
            f.dynamics(y, u0, ydot, A, B, C);
            return ydot;
        };
        auto bfun = [&cov, &y0, &u0](const Vector6d& b)
        {
            Imu3DFactorCostFunction f(0, b, cov);
            Vector9d ydot;
            Matrix9d A;
            Eigen::Matrix<double, 9, 6> B;
            Eigen::Matrix<double, 9, 6> C;
            f.dynamics(y0, u0, ydot, A, B, C);
            return ydot;
        };
        auto ufun = [&cov, &b0, &y0](const Vector6d& u)
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

    auto fun = [&cov, &meas, &t, &y0](const Vector6d& b0)
    {
        Imu3DFactorCostFunction f(0, b0, cov);
        for (int i = 0; i < meas.size(); i++)
        {
            f.integrate(t[i], meas[i]);
        }
        return f.y_;
    };
    JFD = calc_jac(fun, b0, nullptr, nullptr, boxminus, nullptr);

//    cout << "J:\n" << J << "\n\n";
//    cout << "JFD:\n" << JFD << "\n\n";
    EXPECT_LE((J-JFD).array().square().sum(), 1e-3);
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
    Matrix6d P = Matrix6d::Identity() * 1e-8;
    problem.AddResidualBlock(new XformNodeFactorAutoDiff(new XformNodeFactorCostFunction(x.col(1), P)), NULL, xhat.data()+7);


    Solver::Options options;
    options.max_num_iterations = 25;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = false;
    Solver::Summary summary;

//    cout << "xhat0\n" << xhat.transpose() << endl;
//    cout << "bhat0\n" << bhat.transpose() << endl;

    ceres::Solve(options, &problem, &summary);
    double error = (b - bhat).norm();

//    cout << summary.FullReport();
//    cout << "x\n" << x.transpose() << endl;
//    cout << "xhat0\n" << xhat.transpose() << endl;
//    cout << "b\n" << b.transpose() << endl;
//    cout << "bhat\n" << bhat.transpose() << endl;
//    cout << "e " << error << endl;
//    Vector9d residual0, residualf;
//    (*factor)(xhat.data(), xhat.data()+7, vhat.data(), vhat.data()+3, bhat.data(), residual0.data());
//    (*factor)(xhat.data(), xhat.data()+7, vhat.data(), vhat.data()+3, b.data(), residualf.data());

//    cout << "\nresidual0: " << residual0.transpose() << endl;
//    cout << "residualf: " << residualf.transpose() << endl;
//    cout << "\ny: " << factor->y_.transpose() << endl;
//    cout << "y+dy: " << boxplus(factor->y_, factor->J_ *(bhat - factor->bhat_)).transpose() << endl;
//    cout << "\nP: \n" << factor->P_ << endl;
    EXPECT_LE(error, 0.1);
}

TEST(Imu3D, MultiWindow)
{
    Simulator multirotor(false);
    multirotor.load("../lib/multirotor_sim/params/sim_params.yaml");

    const int N = 1000;

    Vector6d b, bhat;
    b.block<3,1>(0,0) = multirotor.get_accel_bias();
    b.block<3,1>(3,0) = multirotor.get_gyro_bias();
    bhat.setZero();
    bhat = b;

    Problem problem;

    Eigen::Matrix<double, 7, N+1> xhat, x;
    Eigen::Matrix<double, 3, N+1> vhat, v;

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

    std::vector<Imu3DFactorCostFunction*> factors;
    factors.push_back(new Imu3DFactorCostFunction(0, bhat, multirotor.get_imu_noise_covariance()));

    // Integrate for N frames
    int node = 0;
    Imu3DFactorCostFunction* factor = factors[node];
    std::vector<double> t;
    t.push_back(multirotor.t_);
    while (node < N)
    {
        t.push_back(multirotor.t_);
        multirotor.run();
        factor->integrate(multirotor.t_, multirotor.get_imu_prev());
        multirotor.get_measurements(meas_list);
        bool new_node = false;
        for (auto it = meas_list.begin(); it != meas_list.end(); it++)
        {
            switch(it->type)
            {
            case Simulator::FEAT:
                // simulate a camera measurement
                new_node = true;
                break;
            default:
                break;
            }
        }

        if (new_node)
        {
            node += 1;

            // estimate next node pose and velocity with IMU preintegration
            factor->estimate_xj(xhat.data()+7*(node-1), vhat.data()+3*(node-1), xhat.data()+7*(node), vhat.data()+3*(node));
            // Calculate the Information Matrix of the IMU factor
            factor->finished();

            // Save off True Pose and Velocity for Comparison
            x.col(node) = multirotor.get_pose().arr_;
            v.col(node) = multirotor.dyn_.get_state().segment<3>(dynamics::VX);

            // Declare the new parameters used for this new node
            problem.AddParameterBlock(xhat.data()+7*node, 7, new XformAutoDiffParameterization());
            problem.AddParameterBlock(vhat.data()+3*node, 3);

            // Add IMU factor to graph
            problem.AddResidualBlock(new Imu3DFactorAutoDiff(factor), NULL, xhat.data()+7*(node-1), xhat.data()+7*node, vhat.data()+3*(node-1), vhat.data()+3*node, bhat.data());

            // Start a new Factor
            factors.push_back(new Imu3DFactorCostFunction(multirotor.t_, bhat, multirotor.get_imu_noise_covariance()));
            factor = factors[node];
        }
    }




    Matrix6d P = Matrix6d::Identity() * 0.1;
    problem.AddResidualBlock(new XformNodeFactorAutoDiff(new XformNodeFactorCostFunction(x.col(1), P)), NULL, xhat.data()+7);


    Solver::Options options;
    options.max_num_iterations = 100;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = true;
    Solver::Summary summary;
    ofstream truth_file("Imu3d.MultiWindow.truth.log", ios::out);
    ofstream est_file("Imu3d.MultiWindow.est.log", ios::out);

    cout.flush();

//    cout << "xhat0\n" << xhat.transpose() << endl;
//    cout << "bhat0\n" << bhat.transpose() << endl;

//    ceres::Solve(options, &problem, &summary);
    double error = (b - bhat).norm();

    cout << summary.FullReport();
    cout << "x\n" << x.transpose() << endl;
    cout << "xhat0\n" << xhat.transpose() << endl;
    cout << "b\n" << b.transpose() << endl;
    cout << "bhat\n" << bhat.transpose() << endl;
    cout << "e " << error << endl;
    EXPECT_LE(error, 0.10);

    Eigen::Matrix<double, 9, N> final_residuals;

    cout << "R\n";
    for (int node = 1; node <= N; node++)
    {
        (*factors[node-1])(xhat.data()+7*(node-1), xhat.data()+7*node,
                         vhat.data()+3*(node-1), vhat.data()+3*node,
                         bhat.data(),
                         final_residuals.data()+9*node);
        cout << final_residuals.col(node-1).transpose() << "\n";

    }
    cout << endl;


    for (int i = 0; i <= N; i++)
    {
        truth_file.write((char*)&t[i],sizeof(double));
        truth_file.write((char*)(x.data()+7*i),sizeof(double)*7);
        truth_file.write((char*)(v.data()+3*i),sizeof(double)*3);
        est_file.write((char*)&t[i],sizeof(double));
        est_file.write((char*)(xhat.data()+7*i),sizeof(double)*7);
        est_file.write((char*)(vhat.data()+3*i),sizeof(double)*3);
    }
    truth_file.close();
    est_file.close();
}
