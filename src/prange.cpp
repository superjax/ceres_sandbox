#include <fstream>

#include <ceres/ceres.h>
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <gtest/gtest.h>

#include "geometry/xform.h"
#include "multirotor_sim/simulator.h"
#include "multirotor_sim/controller.h"
#include "multirotor_sim/satellite.h"
#include "utils/estimator_wrapper.h"

#include "factors/pseudorange.h"


using namespace multirotor_sim;
using namespace ceres;
using namespace Eigen;
using namespace xform;


TEST (Pseudorange, TestCompile)
{
    GTime t;
    Vector2d rho;
    Satellite sat(1);
    Vector3d rec_pos;
    Matrix2d cov;
    PseudorangeCostFunction prange_factor(t, rho, sat, rec_pos, cov);
}
