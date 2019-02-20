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
#include "multirotor_sim/estimator_wrapper.h"
#include "utils/logger.h"
#include "test_common.h"

#include "factors/SE3.h"
#include "factors/imu_3d.h"
#include "factors/switch_pseudorange.h"
#include "factors/pseudorange.h"
#include "factors/carrier_phase.h"
#include "factors/clock_bias_dynamics.h"


using namespace multirotor_sim;
using namespace Eigen;
using namespace xform;

class TrajectorySim : public EstimatorBase
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    int n = 0;
    Simulator sim;
    static const int N = 100;
    Eigen::Matrix<double, 7, N> xhat0, xhat, x;
    Eigen::Matrix<double, 3, N> vhat0, vhat, v;
    Eigen::Matrix<double, 2, N> dt_hat0, dt_hat, dt;
    Eigen::Matrix<double, -1, -1> shat0, shat, s;
    std::vector<double> t;
    Xformd x_e2n_hat;
    Vector6d b, bhat;
    Matrix2d dt_cov = Vector2d{1e-5, 1e-6}.asDiagonal();

    std::vector<Vector3d, aligned_allocator<Vector3d>> measurements;
    std::vector<Matrix3d, aligned_allocator<Matrix3d>> cov;
    std::vector<GTime> gtimes;

    ceres::Problem problem;
    bool new_node;
    std::vector<Imu3DFunctor*> factors;
    Imu3DFunctor* factor;
    std::vector<std::function<void(void)>> new_node_funcs;

    GTime t0;
    std::vector<double> Phi0;

    double switch_weight;
    double switch_dyn_weight;

    double error0;

    TrajectorySim(const string &yaml_file);
    void init(const std::string& yaml_file);
    void init_residual_blocks();
    void imuCallback(const double& t, const Vector6d& z, const Matrix6d& R) override;
    void rawGnssCallback(const GTime& t, const VecVec3& z, const VecMat3& R, std::vector<Satellite>& sats, const std::vector<bool>& slip) override;
    void init_estimator();
    void fix_origin();
    void addImuFactor();
    void initNodePostionFromPointPos();
    void addPseudorangeFactors();
    void addPseudorangeFactorsWithClockDynamics();
    void addSwitchingPseudorangeFactors();
    void addCarrierPhaseFactors();
    void addNodeCB(std::function<void(void)> cb);
    void run();
    void solve();
    void log(string filename);
    double final_error();


};

