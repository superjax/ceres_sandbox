#include <fstream>

#include "test_common.h"

using namespace std;

std::string imu_only_sim()
{
    YAML::Node yaml;
    yaml = YAML::LoadFile("../lib/multirotor_sim/params/sim_params.yaml");
    ofstream tmp("/tmp/ceres_sandbox.tmp.yaml");
    yaml["imu_enabled"] =  true;
    yaml["alt_enabled"] =  false;
    yaml["mocap_enabled"] =  false;
    yaml["vo_enabled"] =  false;
    yaml["camera_enabled"] =  false;
    yaml["gnss_enabled"] =  false;
    yaml["raw_gnss_enabled"] =  false;
    tmp << yaml;
    tmp.close();
    return "/tmp/ceres_sandbox.tmp.yaml";
}

std::string raw_gps_yaml_file()
{
    YAML::Node node;
    node = YAML::LoadFile("../lib/multirotor_sim/params/sim_params.yaml");
    ofstream tmp("/tmp/ceres_sandbox.tmp.yaml");
    node["imu_enabled"] =  true;
    node["alt_enabled"] =  false;
    node["mocap_enabled"] =  false;
    node["vo_enabled"] =  false;
    node["camera_enabled"] =  false;
    node["gnss_enabled"] =  false;
    node["raw_gnss_enabled"] =  true;
    node["ephemeris_filename"] =  "../lib/multirotor_sim/sample/eph.dat";

    tmp << node;
    tmp.close();
    return "/tmp/ceres_sandbox.tmp.yaml";
}

std::string raw_gps_multipath_yaml_file()
{
    YAML::Node node;
    node = YAML::LoadFile("../lib/multirotor_sim/params/sim_params.yaml");
    ofstream tmp("/tmp/ceres_sandbox.tmp.yaml");
    node["imu_enabled"] =  true;
    node["alt_enabled"] =  false;
    node["mocap_enabled"] =  false;
    node["vo_enabled"] =  false;
    node["camera_enabled"] =  false;
    node["gnss_enabled"] =  false;
    node["raw_gnss_enabled"] =  true;
    node["ephemeris_filename"] =  "../lib/multirotor_sim/sample/eph.dat";

    node["multipath_prob"] = 0.1;
    node["cycle_slip_prob"] = 0.0;

    tmp << node;
    tmp.close();
    return "/tmp/ceres_sandbox.tmp.yaml";
}

std::string mocap_yaml_file()
{
    YAML::Node node;
    node = YAML::LoadFile("../lib/multirotor_sim/params/sim_params.yaml");
    ofstream tmp("/tmp/ceres_sandbox.tmp.yaml");
    node["imu_enabled"] =  true;
    node["alt_enabled"] =  false;
    node["mocap_enabled"] =  true;
    node["vo_enabled"] =  false;
    node["camera_enabled"] =  false;
    node["gnss_enabled"] =  false;
    node["raw_gnss_enabled"] =  false;

    tmp << node;
    tmp.close();
    return "/tmp/ceres_sandbox.tmp.yaml";
}

