#pragma once

#include <vector>
#include <Eigen/Core>

#include "utils/jac.h"

using namespace std;
using namespace Eigen;


template <typename T>
class Camera
{
public:
    typedef Matrix<T,2,1> Vec2;
    typedef Matrix<T,2,2> Mat2;
    typedef Matrix<T,3,1> Vec3;
    typedef Matrix<T,5,1> Vec5;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Camera(const Vec2& f, const Vec2& c, const Vec5& d, const T& s) :
        focal_len_(f.data()),
        cam_center_(c.data()),
        s_(s),
        distortion_(d.data())
    {}

    Camera(const Vec2& f, const Vec2& c, const Vec5& d, const T& s, const Vector2d& size) :
        focal_len_(f.data()),
        cam_center_(c.data()),
        s_(s),
        distortion_(d.data())
    {
        setSize(size);
    }

    Camera(const T* f, const T* c, const T* d, const T* s) :
        focal_len_(f),
        cam_center_(c),
        s_(*s),
        distortion_(d)
    {}

    void setSize(const Vector2d& size)
    {
        image_size_ = size;
    }

    void unDistort(const Vec2& pi_u, Vec2& pi_d) const
    {
        const T k1 = distortion_(0);
        const T k2 = distortion_(1);
        const T p1 = distortion_(2);
        const T p2 = distortion_(3);
        const T k3 = distortion_(4);
        const T x = pi_u.x();
        const T y = pi_u.y();
        const T xy = x*y;
        const T xx = x*x;
        const T yy = y*y;
        const T rr = xx*yy;
        const T r4 = rr*rr;
        const T r6 = r4*rr;


        // https://docs.opencv.org/2.4/doc/tutorials/calib3d/camera_calibration/camera_calibration.html
        const T g =  1.0 + k1 * rr + k2 * r4 + k3*r6;
        const T dx = 2.0 * p1 * xy + p2 * (rr + 2.0 * xx);
        const T dy = 2.0 * p2 * xy + p1 * (rr + 2.0 * yy);

        pi_d.x() = g * (x + dx);
        pi_d.y() = g * (y + dy);
    }

    void Distort(const Vec2& pi_d, Vec2& pi_u, double tol=1e-6) const
    {
        pi_u = pi_d;
        Vec2 pihat_d;
        Mat2 J;
        Vec2 e;
        T prev_e = (T)1000.0;
        T enorm = (T)0.0;

        static const int max_iter = 50;
        int i = 0;
        while (i < max_iter)
        {
            unDistort(pi_u, pihat_d);
            e = pihat_d - pi_d;
            enorm = e.norm();
            if (enorm <= tol || prev_e < enorm)
                break;
            prev_e = enorm;

            distortJac(pi_u, J);
            pi_u = pi_u - J*e;
            i++;
        }


        if ((pi_u.array() != pi_u.array()).any())
        {
            int debug = 1;
        }
    }

    void distortJac(const Vec2& pi_u, Mat2& J) const
    {
        const T k1 = distortion_(0);
        const T k2 = distortion_(1);
        const T p1 = distortion_(2);
        const T p2 = distortion_(3);
        const T k3 = distortion_(4);

        const T x = pi_u.x();
        const T y = pi_u.y();
        const T xy = x*y;
        const T xx = x*x;
        const T yy = y*y;
        const T rr = xx+yy;
        const T r = sqrt(rr);
        const T r4 = rr*rr;
        const T r6 = rr*r4;
        const T g =  (T)1.0 + k1 * rr + k2 * r4 + k3*r6;
        const T dx = (x + ((T)2.0*p1*xy + p2*(rr+(T)2.0*xx)));
        const T dy = (y + (p1*(rr+(T)2.0*yy) + (T)2.0*p2*xy));

        const T drdx = x / r;
        const T drdy = y / r;
        const T dgdx = k1*(T)2.0*r*drdx + (T)4.0*k2*rr*r*drdx + (T)6.0*k3*r4*r*drdx;
        const T dgdy = k1*(T)2.0*r*drdy + (T)4.0*k2*rr*r*drdy + (T)6.0*k3*r4*r*drdy;

        J << /* dxbar/dx */ ((T)1.0 + ((T)2.0*p1*y + p2*((T)2.0*r*drdx + (T)4.0*x)))*g + dx*dgdx,
             /* dxbar/dy */ ((T)2.0*p1*x + p2*(T)2.0*r*drdy)*g + dx*dgdy,
             /* dybar/dx */ (p1*(T)2.0*r*drdx+(T)2.0*p2*y)*g + dy*dgdx,
             /* dybar/dy */ ((T)1.0 + (p1*((T)2.0*r*drdy + (T)4.0*y) + (T)2.0*p2*x))*g + dy*dgdy;

        if ((J.array() != J.array()).any())
        {
            int debug = 1;
        }
    }


    void pix2intrinsic(const Vec2& pix, Vec2& pi) const
    {
        const T fx = focal_len_.x();
        const T fy = focal_len_.y();
        const T cx = cam_center_.x();
        const T cy = cam_center_.y();
        pi << (1.0/fx) * (pix.x() - cx - (s_/fy) * (pix.y() - cy)),
                (1.0/fy) * (pix.y() - cy);
    }

    void intrinsic2pix(const Vec2& pi, Vec2& pix) const
    {
        const T fx = focal_len_.x();
        const T fy = focal_len_.y();
        const T cx = cam_center_.x();
        const T cy = cam_center_.y();
        pix << fx*pi.x() + s_*pi.y() + cx,
               fy*pi.y() + cy;
    }

    void proj(const Vec3& pt, Vec2& pix) const
    {
        const T pt_z = pt(2);
        Vec2 pi_d;
        Vec2 pi_u = (pt.template segment<2>(0) / pt_z);
        Distort(pi_u, pi_d);
        intrinsic2pix(pi_d, pix);
    }

    inline bool check(const Vector2d& pix) const
    {
        return !((pix.array() > image_size_.array()).any()|| (pix.array() < 0).any());
    }

    void invProj(const Vec2& pix, const T& depth, Vec3& pt) const
    {
        Vec2 pi_d, pi_u;
        pix2intrinsic(pix, pi_d);
        unDistort(pi_d, pi_u);
        pt.template segment<2>(0) = pi_u;
        pt(2) = 1.0;
        pt *= depth / pt.norm();
    }

    Map<const Vec2> focal_len_;
    Map<const Vec2> cam_center_;
    Map<const Vec5> distortion_;
    const T& s_;
    Vector2d image_size_;

private:
    const Matrix2d I_2x2 = Matrix2d::Identity();
};

