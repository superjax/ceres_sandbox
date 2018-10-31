# Ceres Solver Sandbox

I wanted to learn the [ceres solver](http://ceres-solver.org/).  There are a lot of great tutorials on their wiki, but I felt like I needed some practical experience with some simple problems before I jumped into full-blown SLAM.

So, I broke the problem into little peices and wrote a bunch of simple cases to explore the capabilities of Ceres.  Each section is written as a series of unit tests in the [gtest](https://github.com/google/googletest) framework.

Borrowing off the _factor graph_ mindset (I'm from a SLAM background), I organized all my cost functions into _factors_, found in `include/factors`

 * [Position1D](#position1d)
    * [Position1D.AveragePoints](#position1daveragepoints)
    * [Position1D.AveragePointsWithParameterBlock](#position1daveragepointswithparameterblock)
    * [Robot1D.SLAM](#robot1dslam)
 * [Position3D](#position3d)
    * [Position3D.AveragePoints](#position3daveragepoints)
 * [Attitude3D](#attitude3d)
    * [Attitude3d.Check*](#attitude3dcheck)
    * [Attitude3d.AverageAttitude](#attitude3daverageattitude)
    * [Attitude3d.AverageAttitudeAutoDiff](#attitude3daverageattitudeautodiff)
 * [Pose3D](#pose3d)
    * [Pose3D.AveragePoseAutoDiff](#pose3daverageposeautodiff)
    * [Pose3D.GraphSLAM](#pose3dgraphslam)
 * [Imu1D](#imu1d)
    * [IMU.1DRobotSingleWindow](#imu1drobotsinglewindow)
    * [IMU.1DRobotLocalization](#imu1drobotlocalization)
    * [IMU.1DRobotSLAM](#imu1drobotslam)
    * [IMU.dydb](#imudydb)
 * [Imu3D](#imu3d)
    * [Imu3D.CheckDynamicsJacobians](#imu3dcheckdynamicsjacobians)
    * [Imu3D.CheckBiasJacobians](#imu3dcheckbiasjacobians)
    * [Imu3D.SingleWindow](#imu3dsinglewindow)
    * [Imu3D/MultiWindow](#imu3dmultiwindow)
 * [TimeOffset](#timeoffset)
    * [TimeOffset.1DRobotSLAM](#timeoffset1drobotslam)
    * [TimeOffset.3DMultirotorPoseGraph](#timeoffset3dmultirotorposegraph)
    * [IMU.3DRobotSLAM](#imu3drobotslam)
 * [Camera](#cameracpp)
    * [Camera.Intrinsics_Calibration](#cameraintrinsics_calibration)


# Position1D
The first set of unit tests looks at the simplest problems.

## Position1D.AveragePoints
This finds the average of a bunch of 1D samples.  It's a little like opening a piggy bank with a jackhammer, but we gotta start somewhere.  Uses the `Position1DFactor` with analytical (trivial) jacobians.

## Position1D.AveragePointsWithParameterBlock
This is exactly like `Position1D.AveragePoints`, but it adds a (redundant) parameter block.  Just to test how `problem.AddParameterBlock` works.

## Robot1D.SLAM
This performs 1D SLAM (haha!).  There is a 1D robot which takes a 1m step along the real line, 7 times.  Each step, he gets a (noisy) range measurement to 3 landmarks (also on the real line).  It's a pretty trivial problem, but I wanted to explore how to handle multiple factors, and handle measurement variance in my factors.

Uses `Transform1d` - the transform between each step with associated variance and `Range1dFactor` - the range to a landmark and associated variance.


# Position3D
## Position3D.AveragePoints
This finds the average of a bunch of 1D samples.  Again, overkill, but uses the `Position3DFactor` with analytical jacobians.  I also used this example to explore how Eigen interacts with Ceres.


# Attitude3D
The second set of unit tests looks at attitude, and the `LocalParameterization` functionality in ceres.

## Attitude3d.Check*
In writing the `QuatFactor` and `QuatParameterization` classes, I wanted to make sure that the associated `Plus`, `Evaluate` and `ComputeJacobian` functions were correct.  These were pretty straight-forward, except the jacobian of the `QuatFactor`.  That was terrible.  Anyway, these are good for sanity checking the operations being performed by the factor and local parameterization.

## Attitude3d.AverageAttitude
This takes a quaternion and creates 1000 samples normally distributed about this quaternion (using the tangent space to come up with the samples).  Then, I use ceres solver to recover the mean.  Probably the stupidest way to take a mean of a data set, but it exercises the use of a local parameterization, and non-trivial analytic jacobians over my factor.

## Attitude3d.AverageAttitudeAutoDiff
Finds the average attitude of a sample of 1000 attitude measurements using the autodiff functionality for both the localparameterization and cost function.  I also used my templated `Quat` library and let Ceres auto-diff through my library (based on Eigen).

This example led to an interesting result.  I spent something like 6 hours deriving the analytical jacobian for the previous example, and found essentially _zero_ performance increase (In fact, the auto-diff version is typically a little bit faster).  While I feel a little sheepish for thinking that I could out-optimize the compiler on this once, fewer jacobians over manifolds for me is a good thing.  After this result I pretty much stopped using analytical jacobians at all and let the auto-differentiation engine do its magic.

# Pose3D
Next, I figured I could do some pose-graph SLAM in SE3

## Pose3D.AveragePoseAutoDiff
As before, just to exercise auto differentiation over a tangent space, I used ceres to find the average of a set of 1000 random `Xform` samples. (using my templated `Xform` library to represent members of homogeneous transforms)

## Pose3D.GraphSLAM
This is the most basic kind of non-trivial SLAM.  We have a bunch of nodes, and edges between nodes.  We also have loop-closures so that the graph is over-constrained and require optimization to find the maximum-likelihood configuration of all the nodes and edges.

I'm using full 6DOF edges and nodes in this graph and the first node is fixed at the origin.

I'm also using auto-differentiated Factors and Local Parameterizations

# Imu1D
Next, I wanted to look into estimating motion of a robot using IMU inputs.  The following examples use a simulated robot that moves at a constant rate along the real line, and but has a noisy IMU with a constant bias.

## IMU.1DRobotSingleWindow
This example looks at a single preintegration window with the origin pose set constant and a direct measurement of the second pose.  Basically, this shows that the unknown IMU bias can be inferred using my `Imu1DFactorAutoDiff` factor.

## IMU.1DRobotLocalization
This example looks at 100 preintegration intervals with with a common constant unknown bias.  The origin is set constant, and the final pose is given a very strong position and velocity measurement.  The IMU bias is inferred using my `IMU1DFactorAutoDiff` factor.

## IMU.1DRobotSLAM
Using the `Range1dFactor` with range measurements to several landmarks also on the real line and the `IMU1dFactor` to perform 1D SLAM.

## IMU.dydb
In writing the `IMU1dFactor`, I had to figure out the jacobian to map changes in bias to changes in the measurement.  This test proves that the jacobian I cam up with is right.

# Imu3D
The biggest reason I did all this was to help me in my work on SLAM with autonomous agents.  Often, I have IMU measurements onboard, but these occur at a very high rate.  The goal of these examples is to preintegrate these IMU measurements and estimate biases while doing SLAM.

## Imu3D.CheckDynamicsJacobians
This examples simply checks the jacobians of the dynamics in the `Imu3DFactorCostFunction` class

## Imu3D.CheckBiasJacobians
This example checks the jacobian used to modify the preintegrated measurement given a change in bias.

## Imu3D.SingleWindow
This preintegrates a single window and estimates the IMU bias given a fixed origin and a measurement of the final pose.

## Imu3D/MultiWindow
This example simulates the flight of a multirotor given measurements of pose at regular intervals.  IMU is preintegrated at 250Hz while Pose measurements are supplied at 5Hz.  The constant IMU biases are estimated over the interval.  The results of this study can be visualized using the `Imu3DMultiWindowPlot.m` matlab script.

# TimeOffset
Another problem in Robotics is time synchronization between different sensors.  The following examples show how to estimate this time offset.

## TimeOffset.1DRobotSLAM
This example is a 1D robot performing SLAM with IMU preintegration as in the `IMU.1DRobotSLAM` example, except that I also have a measurement of the position of each node with a small time delay.  I use the `Position1dFactorWithTimeOffset` factor to estimate this delay.

## TimeOffset.3DMultirotorPoseGraph
This example is of a multirotor flying waypoints with a lagged position and attitude measurement (as experienced by a motion capture system).  The goal is to estimate this offset, IMU bias and the location of all poses simultaneously.  The results of this example can be visualized by running the `TimeOffsetMultiWindowPlot.m` matlab script.

## IMU.3DRobotSLAM
This example is the full SLAM problem - a 3D rigid body moves in space, and has bearing measurements to several landmarks.  Performs SLAM while inferring IMU biases.

# Camera
Next, I wanted to use the ceres solver to deal with the projection associated with a pinhole camera model.

## Camera.Intrinsics_Calibration
This example simulates the calibration of a pinhole camera.  A 3D rigid body gets simulated pixel measurements to known landmarks in the camera FOV. The camera intrinsics are estimated.


# Building The Code
There aren't any real linux-specific dependencies that I know of, but I use linux almost exclusively and do not know if this will work on Windows or Mac. You do need Eigen `sudo apt install libeigen3-dev` and the ceres sovler, [installation](http://ceres-solver.org/installation.html).  I would recommend building the latetest stable release from source and be sure to grab the suitesparse and cxsparse dependencies. This development took place with ceres 1.14.0 and eigen 3.3.4.  I noticed that Eigen threw alignment errors with the `master` branch, which I found odd.  You'll also need gtest [installation](https://www.eriksmistad.no/getting-started-with-google-test-on-ubuntu/).  

This project depends on my templated `geometry` library for homogeneous transforms and quaternions as well as my C++ multirotor simulator I developed with @jerelbn.  (These are included as submodules).

Once you've got all the libraries installed, it's a simple CMake build.

``` bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j -l
```

# Running The Examples
All the code in this repo is organized into a big unit test suite.  The tests usuall all pass, but not always, and if they don't they are really close.  Just run the `ceres_tests` executable.  Being a Gtest suite, it has a number of command-line arguments that it supports.  Try `-h` to see the available command-line options.

```
./ceres_tests
```