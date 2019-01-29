#include <Eigen/Core>
#include <ceres/ceres.h>

#include "geometry/cam.h"
#include "geometry/xform.h"

using namespace Eigen;
using namespace xform;

class CamFunctor
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    CamFunctor(const Vector2d& pix, const Matrix2d& cov, const Vector2d& img_size)
    {
        pix_ = pix;
        size_ = img_size;
        Xi_ = cov.inverse().llt().matrixL().transpose();
    }

    template <typename T>
    bool operator()(const T* _ptw, const T* _xw2c, const T* _f, const T* _c, const T* _s, const T* _d, T* _res) const
    {
        typedef Matrix<T,3,1> Vec3;
        typedef Matrix<T,2,1> Vec2;

        Camera<T> cam(_f, _c, _d, _s);
        Map<const Vec3> pt_w(_ptw); // point in the world frame
        Xform<T> x_w2c(_xw2c); // transform from world to camera
        Map<Vec2> res(_res); // residuals

        Vec3 pt_c = x_w2c.transformp(pt_w); // point in the camera frame
        Vec2 pixhat; // estimated pixel location
        cam.proj(pt_c, pixhat);

        res = Xi_ * (pix_ - pixhat);
        return true;
    }

private:
    Vector2d pix_;
    Vector2d size_;
    Matrix2d Xi_;
};

typedef ceres::AutoDiffCostFunction<CamFunctor, 2, 3, 7, 2, 2, 1, 5> CamFactorAD;
