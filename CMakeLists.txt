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
    lib/multirotor_sim/include
    lib/geometry/include
    lib/utils/include
    lib/factors/include
    ${CERES_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS})

add_library(utils
    lib/utils/src/jac.cpp
    lib/utils/src/robot1d.cpp
    lib/utils/src/estimator_wrapper.cpp)

add_executable(dev_test
    src/dev_test.cpp)
target_link_libraries(dev_test
    utils
    geometry
    multirotor_sim
    gtest
    gtest_main
    ${GTEST_LIBRARIES}
    ${CERES_LIBRARIES}
)


add_executable(ceres_tests
    src/position1d.cpp
    src/position3d.cpp
    src/attitude.cpp
    src/pose.cpp
    src/imu1d.cpp
    src/imu3d.cpp
    src/time_offset.cpp
    src/camera.cpp
    src/pseudorange.cpp
    src/control.cpp
    src/carrier_phase.cpp
)

target_link_libraries(ceres_tests
    utils
    geometry
    multirotor_sim
    gtest
    gtest_main
    ${GTEST_LIBRARIES}
    ${CERES_LIBRARIES}
)

