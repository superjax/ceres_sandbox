#!/bin/bash

cd ..
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j12 -l12
./test_attitude
./test_camera
./test_carrier_phas
./test_control
./test_imu1d
./test_imu3d
./test_pose
./test_position1d
./test_position3d
./test_pseudorange
./test_switch
./test_time_offset
