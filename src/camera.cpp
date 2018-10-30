#include <gtest/gtest.h>
#include <ceres/ceres.h>

#include "geometry/xform.h"
#include "cam.h"
#include "factors/camera.h"
#include "factors/SE3.h"
#include "multirotor_sim/utils.h"

using namespace Eigen;
using namespace ceres;
using namespace xform;

TEST(Camera, Proj_InvProj)
{
    Vector2d focal_len{250.0, 250.0};
    Vector2d cam_center{320.0, 240.0};
    Vector2d img_size{640, 480};
    Vector5d distortion = (Vector5d() << -0.28340811, 0.07395907, 0.00019359, 1.76187114e-05, 0.0).finished();
    double s = 0.0;
    Camera<double> cam(focal_len, cam_center, distortion, s, img_size);

    Vector3d pt{1.0, 0.5, 2.0};
    double depth = pt.norm();
    Vector2d pix;
    Vector3d pt2;

    cam.proj(pt, pix);
    cam.invProj(pix, depth, pt2);

    EXPECT_NEAR(pt.x(), pt2.x(), 1e-8);
    EXPECT_NEAR(pt.y(), pt2.y(), 1e-8);
    EXPECT_NEAR(pt.z(), pt2.z(), 1e-8);
}

TEST(Camera, Distort_UnDistort)
{
    Vector2d focal_len{250.0, 250.0};
    Vector2d cam_center{320.0, 240.0};
    Vector2d img_size{640, 480};
    Vector5d distortion = (Vector5d() << -0.28340811, 0.07395907, 0.00019359, 1.76187114e-05, 0.0).finished();
    double s = 0.0;
    Camera<double> cam(focal_len, cam_center, distortion, s, img_size);

    Vector2d pix_d, pix_u;
    Vector2d pix_d2, pi_d2;
    Vector3d pt{1.0, 0.5, 2.0};

    cam.proj(pt, pix_d);
    Vector2d pi_d, pi_u;

    cam.pix2intrinsic(pix_d, pi_d);
    cam.Distort(pi_d, pi_u);
    cam.unDistort(pi_u, pi_d2);
    cam.intrinsic2pix(pi_d2, pix_d2);

    EXPECT_NEAR(pix_d2.x(), pix_d.x(), 1e-3);
    EXPECT_NEAR(pix_d2.y(), pix_d.y(), 1e-3);
}

TEST(Camera, UnDistort_Distort)
{
    Vector2d focal_len{250.0, 250.0};
    Vector2d cam_center{320.0, 240.0};
    Vector2d img_size{640, 480};
    Vector5d distortion = (Vector5d() << -0.28340811, 0.07395907, 0.00019359, 1.76187114e-05, 0.0).finished();
    double s = 0.0;
    Camera<double> cam(focal_len, cam_center, distortion, s, img_size);

    Vector2d pix_d, pix_u;
    Vector2d pix_u2, pi_u2;
    Vector3d pt{1.0, 0.5, 2.0};

    cam.proj(pt, pix_d);
    Vector2d pi_d, pi_u;

    cam.pix2intrinsic(pix_u, pi_u);
    cam.unDistort(pi_u, pi_d);
    cam.Distort(pi_d, pi_u2);
    cam.intrinsic2pix(pi_u2, pix_u2);

    EXPECT_NEAR(pix_u2.x(), pix_u.x(), 1e-3);
    EXPECT_NEAR(pix_u2.y(), pix_u.y(), 1e-3);
}

TEST (Camera, DistortJacobian)
{
    Vector2d focal_len{250.0, 250.0};
    Vector2d cam_center{320.0, 240.0};
    Vector2d img_size{640, 480};
    Vector5d distortion = (Vector5d() << -0.28340811, 0.07395907, 0.00019359, 1.76187114e-05, 0.0).finished();
    double s = 0.0;
    Camera<double> cam(focal_len, cam_center, distortion, s, img_size);

    Matrix2d JA, JFD;
    Vector2d x0{0.1, 0.5};

    auto fun = [cam](const MatrixXd& x_u)
    {
        Vector2d x_d;
        cam.unDistort(x_u, x_d);
        return x_d;
    };

    JFD = calc_jac(fun, x0);
    cam.distortJac(x0, JA);
    //        cout << "JA\n" << JA << endl;/
    //        cout << "JFD\n" << JFD << endl;
}

TEST (Camera, Intrinsics_Calibration)
{
    MatrixXd landmarks;
    landmarks.resize(3, 100);
    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < 10; j++)
        {
            landmarks.block<3,1>(0,i*10+j) << (i-5)/10.0, (j-5)/10.0, 0;
        }
    }

    MatrixXd camera_pose;
    camera_pose.resize(7, 9);
    double deg15 = 15.0*M_PI/180.0;
    double deg7 = 7.0*M_PI/180.0;
    camera_pose.col(0) = Xformd(Vector3d{0, 0, -1}, Quatd::Identity()).elements();
    camera_pose.col(1) = Xformd(Vector3d{1, 0, -1}, Quatd::from_euler(deg15, 0, 0)).elements();
    camera_pose.col(2) = Xformd(Vector3d{1, 1, -1}, Quatd::from_euler(deg7, -deg7, 0)).elements();
    camera_pose.col(3) = Xformd(Vector3d{0, 1, -1}, Quatd::from_euler(0, -deg15, 0)).elements();
    camera_pose.col(4) = Xformd(Vector3d{-1, 1, -1}, Quatd::from_euler(-deg7, deg7, 0)).elements();
    camera_pose.col(5) = Xformd(Vector3d{-1, 0, -1}, Quatd::from_euler(-deg15, 0, 0)).elements();
    camera_pose.col(6) = Xformd(Vector3d{-1, -1, -1}, Quatd::from_euler(-deg7, deg7, 0)).elements();
    camera_pose.col(7) = Xformd(Vector3d{0, -1, -1}, Quatd::from_euler(0, deg15, 0)).elements();
    camera_pose.col(8) = Xformd(Vector3d{1, -1, -1}, Quatd::from_euler(deg7, deg7, 0)).elements();

    Vector2d focal_len{250.0, 250.0};
    Vector2d cam_center{320.0, 240.0};
    Vector2d img_size{640, 480};
    Vector5d distortion = (Vector5d() << -0.28340811, 0.07395907, 0.00019359, 1.76187114e-05, 0.0).finished();
    double s = 0.0;
    Camera<double> cam(focal_len, cam_center, distortion, s, img_size);

    Vector2d fhat = focal_len + Vector2d::Random()*25;
    Vector2d chat = cam_center + Vector2d::Random()*25;
    Vector5d dhat = distortion + Vector5d::Random()*1e-3;
//    Vector5d dhat = Vector5d::Random()*1e-3;
//    Vector5d dhat = Vector5d::Zero();
    double shat = s + (rand() % 1000)*1e-6;

    MatrixXd xhat = camera_pose;
    MatrixXd lhat = landmarks;

    Problem problem;
    problem.AddParameterBlock(fhat.data(), 2);
    problem.AddParameterBlock(chat.data(), 2);
    problem.AddParameterBlock(dhat.data(), 5);
    problem.AddParameterBlock(&shat, 1);
    for (int i = 0; i < 100; i++)
    {
        problem.AddParameterBlock(lhat.data()+3*i, 3);
        problem.SetParameterBlockConstant(lhat.data()+3*i);
    }
    for (int i = 0; i < 9; i++)
    {
        problem.AddParameterBlock(xhat.data()+7*i, 7, new XformAutoDiffParameterization());
        problem.SetParameterBlockConstant(xhat.data()+7*i);
    }

    Matrix2d cov = Matrix2d::Identity() * 1e-5;
    for (int i = 0; i < 9; i++)
    {
        Xformd x_w2c(camera_pose.col(i)); // transform from world to camera
        for (int j = 0; j < 100; j++)
        {
            Vector2d pix;
            cam.proj(x_w2c.transformp(landmarks.col(j)), pix);

            if (cam.check(pix))
            {
                problem.AddResidualBlock(new CameraAutoDiff(new CameraFactorCostFunction(pix, cov, cam.image_size_)),
                                         NULL, lhat.data()+3*j, xhat.data()+7*i, fhat.data(), chat.data(), &shat,
                                         dhat.data());
            }
        }
    }

    Solver::Options options;
    options.max_num_iterations = 100;
    options.num_threads = 6;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = true;
    Solver::Summary summary;

    cout << "fhat0\t" << fhat.transpose() << endl;
    cout << "chat0\t" << chat.transpose() << endl;
    cout << "dhat0\t" << dhat.transpose() << endl;
    cout << "shat0\t" << shat << endl;

    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport();

    cout << "fhatf\t" << fhat.transpose() << " : " << focal_len.transpose() << endl;
    cout << "chatf\t" << chat.transpose() << " : " << cam_center.transpose() << endl;
    cout << "dhatf\t" << dhat.transpose() << " : " << distortion.transpose() << endl;
    cout << "shatf\t" << shat << " : " << s << endl;

}
