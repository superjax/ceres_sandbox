#pragma once

#include <functional>
#include <Eigen/Core>

#include "multirotor_sim/estimator_base.h"
#include "geometry/quat.h"
#include "geometry/xform.h"

using namespace Eigen;
using namespace quat;
using namespace xform;

class EstimatorWrapper : public multirotor_sim::EstimatorBase
{
public:
    inline void imuCallback(const double& t, const Vector6d& z, const Matrix6d& R) override { if (imu_cb_) imu_cb_(t, z, R); }
    inline void altCallback(const double& t, const Vector1d& z, const Matrix1d& R) override { if (alt_cb_) alt_cb_(t, z, R); }
    inline void posCallback(const double& t, const Vector3d& z, const Matrix3d& R) override { if (pos_cb_) pos_cb_(t, z, R); }
    inline void attCallback(const double& t, const Quatd& z, const Matrix3d& R) override { if (att_cb_) att_cb_(t, z, R); }
    inline void voCallback(const double& t, const Xformd& z, const Matrix6d& R) override { if (vo_cb_) vo_cb_(t, z, R); }
    inline void featCallback(const double& t, const Vector2d& z, const Matrix2d& R, int id, double depth) override { if (feat_cb_) feat_cb_(t, z, R, id, depth); }
    inline void gnssCallback(const double& t, const Vector6d& z, const Matrix6d& R) override { if (gnss_cb_) gnss_cb_(t, z, R); }
    inline void rawGnssCallback(const GTime& t, const Vector3d& z, const Matrix3d& R, Satellite& sat) override { if (raw_gnss_cb_) raw_gnss_cb_(t, z, R, sat); }

    std::function<void(const double& t, const Vector6d& z, const Matrix6d& R)> imu_cb_;
    std::function<void(const double& t, const Vector1d& z, const Matrix1d& R)> alt_cb_;
    std::function<void(const double& t, const Vector3d& z, const Matrix3d& R)> pos_cb_;
    std::function<void(const double& t, const Quatd& z, const Matrix3d& R)> att_cb_;
    std::function<void(const double& t, const Xformd& z, const Matrix6d& R)> vo_cb_;
    std::function<void(const double& t, const Vector2d& z, const Matrix2d& R, int id, double depth)> feat_cb_;
    std::function<void(const double& t, const Vector6d& z, const Matrix6d& R)> gnss_cb_;
    std::function<void(const GTime& t, const Vector3d& z, const Matrix3d& R, Satellite& sat)> raw_gnss_cb_;

    inline void register_imu_cb(std::function<void(const double& t, const Vector6d& z, const Matrix6d& R)> imu_cb) {imu_cb_ = imu_cb;}
    inline void register_alt_cb(std::function<void(const double& t, const Vector1d& z, const Matrix1d& R)> alt_cb) {alt_cb_ = alt_cb;}
    inline void register_pos_cb(std::function<void(const double& t, const Vector3d& z, const Matrix3d& R)> pos_cb) {pos_cb_ = pos_cb;}
    inline void register_att_cb(std::function<void(const double& t, const Quatd& z, const Matrix3d& R)> att_cb) {att_cb_ = att_cb;}
    inline void register_vo_cb(std::function<void(const double& t, const Xformd& z, const Matrix6d& R)> vo_cb) {vo_cb_ = vo_cb;}
    inline void register_feat_cb(std::function<void(const double& t, const Vector2d& z, const Matrix2d& R, int id, double depth)> feat_cb) {feat_cb_ = feat_cb;}
    inline void register_gnss_cb(std::function<void(const double& t, const Vector6d& z, const Matrix6d& R)> gnss_cb) {gnss_cb_ = gnss_cb;}
    inline void register_raw_gnss_cb(std::function<void(const GTime& t, const Vector3d& z, const Matrix3d& R, Satellite& sat)> raw_gnss_cb) {raw_gnss_cb_ = raw_gnss_cb;}
};
