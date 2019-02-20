cmake_minimum_required(VERSION 2.8.3)
project(ceres_sandbox)

set(CMAKE_CXX_STANDARD 14)
#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfma")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma")

find_package(Ceres REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(GTest REQUIRED)

add_subdirectory(lib/geometry)
add_subdirectory(lib/multirotor_sim)

include_directories(include
    lib
    lib/utils/include
    lib/factors/include
    ${EIGEN3_INCLUDE_DIRS})

add_library(utils
    lib/utils/src/jac.cpp
    lib/utils/src/robot1d.cpp
    lib/utils/src/trajectory_sim.cpp
    src/test_common.cpp)
target_link_libraries(utils multirotor_sim)

function(make_gtest test_name source_file)
    add_executable(${test_name} ${source_file})
    target_link_libraries(${test_name}
        utils
        geometry
        gtest
        gtest_main
        ${MULTIROTOR_SIM_LIBS}
        ${GTEST_LIBRARIES}
        ${CERES_LIBRARIES}
    )
endfunction(make_gtest)

make_gtest(test_position1d      src/position1d.cpp)
make_gtest(test_position3d      src/position3d.cpp)
make_gtest(test_attitude        src/attitude.cpp)
make_gtest(test_pose            src/pose.cpp)
make_gtest(test_imu1d           src/imu1d.cpp)
make_gtest(test_imu3d           src/imu3d.cpp)
make_gtest(test_time_offset     src/time_offset.cpp)
make_gtest(test_camera          src/camera.cpp)
make_gtest(test_pseudorange     src/pseudorange.cpp)
make_gtest(test_control         src/control.cpp)
make_gtest(test_switch          src/switch.cpp)
make_gtest(test_carrier_phase   src/carrier_phase.cpp)


