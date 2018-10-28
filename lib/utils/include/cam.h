#pragma once

#include <vector>
#include <Eigen/Core>
using namespace std;
using namespace Eigen;

static const Matrix2d I_2x2 = Matrix2d::Identity();

template <typename T>
class Camera
{
public:
    typedef Matrix<T,2,1> Vec2;
    typedef Matrix<T,2,2> Mat2;
    typedef Matrix<T,3,1> Vec3;
    typedef Matrix<T,5,1> Vec5;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Camera() :
        focal_len_(buf_),
        cam_center_(buf_+2),
        distortion_(buf_+4)
    {}

    Camera(const Vec2& f, const Vec2& c, const Vec5& d) :
        focal_len_(const_cast<T>(f.data())),
        cam_center_(const_cast<T>(c.data())),
        distortion_(const_cast<T>(d.data()))
    {}

    void setFocalLen(const Vec2& f)
    {
        new (&focal_len_) Map<Vec2>(const_cast<T*>(f.data()));
    }

    void setCamCenter(const Vec2& c)
    {
        new (&cam_center_) Map<Vec2>(const_cast<T*>(c.data()));
    }


    bool proj(const Vec3& pt, Vec2& pix)
    {
        T pt_z = pt(2);
        pix = focal_len_.asDiagonal() * (pt.template segment<2>(0) / pt_z) + cam_center_;
    }


    bool invProj(const Vec2& pix, const T& depth, Vec3& pt)
    {
        pt.template segment<2>(0) = (pix - cam_center_).array() / focal_len_.array();
        pt(2) = 1.0;
        pt *= depth / pt.norm();
    }

    Map<Vec2> focal_len_;
    Map<Vec2> cam_center_;
    Map<Vec5> distortion_;

    T buf_[9];
};
