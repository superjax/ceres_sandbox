cmake_minimum_required(VERSION 2.8.3)
project(ceres_sandbox)

set(CMAKE_CXX_STANDARD 14)

find_package(Ceres REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(GTest REQUIRED)

include_directories(include
    ${CERES_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS})

add_executable(position src/position.cpp)
target_link_libraries(position gtest gtest_main ${GTEST_LIBRARIES} ${CERES_LIBRARIES})

add_executable(attitude src/attitude.cpp)
target_link_libraries(attitude gtest gtest_main ${GTEST_LIBRARIES} ${CERES_LIBRARIES})

add_executable(pose src/pose.cpp)
target_link_libraries(pose gtest gtest_main ${GTEST_LIBRARIES} ${CERES_LIBRARIES})

add_executable(imu src/imu.cpp)
target_link_libraries(imu gtest gtest_main ${GTEST_LIBRARIES} ${CERES_LIBRARIES})

add_executable(lie_test include/lie/tests.cpp)
target_link_libraries(lie_test gtest gtest_main ${GTEST_LIBRARIES} ${CERES_LIBRARIES})

