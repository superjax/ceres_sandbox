#include <fstream>
#include <random>

#include <ceres/ceres.h>
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <gtest/gtest.h>

#include "utils/logger.h"
#include "utils/trajectory_sim.h"
#include "test_common.h"

using namespace multirotor_sim;
using namespace Eigen;
using namespace xform;

TEST (CarrierPhase, Trajectory)
{
    TrajectorySim a(raw_gps_yaml_file());

    a.addNodeCB([&a](){a.initNodePostionFromPointPos();});
    a.addNodeCB([&a](){a.addPseudorangeFactorsWithClockDynamics();});
    a.addNodeCB([&a](){a.addCarrierPhaseFactors();});

    a.run();
    a.solve();
    a.log("/tmp/ceres_sandbox/CarrierPhase.Trajectory.log");

    ASSERT_LE(a.final_error(), a.error0);
}

TEST (CarrierPhase, ImuTrajectory)
{
    TrajectorySim a(raw_gps_yaml_file());
    a.fix_origin();

    a.addNodeCB([&a](){a.addImuFactor();});
    a.addNodeCB([&a](){a.initNodePostionFromPointPos();});
    a.addNodeCB([&a](){a.addPseudorangeFactorsWithClockDynamics();});
    a.addNodeCB([&a](){a.addCarrierPhaseFactors();});

    a.run();
    a.solve();
    a.log("/tmp/ceres_sandbox/CarrierPhase.ImuTrajectory.log");

    ASSERT_LE(a.final_error(), a.error0);
}


