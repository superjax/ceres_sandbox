cmake_minimum_required(VERSION 2.8.3)
project(ceres_sandbox)

set(CMAKE_CXX_STANDARD 14)

find_package(Ceres REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(GTest REQUIRED)

include_directories(include
    ${CERES_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS})

add_executable(ceres_tests
    include/lie/tests.cpp
    src/position_1d.cpp
    src/position_3d.cpp
    src/attitude.cpp
    src/pose.cpp
    src/imu_1d.cpp
    src/imu_3d.cpp
)
target_link_libraries(ceres_tests gtest gtest_main ${GTEST_LIBRARIES} ${CERES_LIBRARIES})
