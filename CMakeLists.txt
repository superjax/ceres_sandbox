cmake_minimum_required(VERSION 2.8.3)
project(ceres_sandbox)

find_package(Ceres REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(GTest REQUIRED)

include_directories(include
    ${CERES_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS})

add_executable(position src/position.cpp)
target_link_libraries(position gtest gtest_main ${GTEST_LIBRARIES} ${CERES_LIBRARIES})
