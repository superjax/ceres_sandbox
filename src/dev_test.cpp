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
#include "test_common.h"

#include "factors/pseudorange.h"
#include "factors/SE3.h"


using namespace multirotor_sim;
using namespace Eigen;
using namespace xform;



