# Ceres Solver Sandbox

I wanted to learn the [ceres solver](http://ceres-solver.org/).  There are a lot of great tutorials on their wiki, but I felt like I needed some practical experience with some simple problems before I jumped into full-blown SLAM.

So, I broke the problem into little peices and wrote a bunch of simple cases to explore the capabilities of Ceres.  Each section is written as a series of unit tests in the [gtest](https://github.com/google/googletest) framework.

Borrowing off the _factor graph_ mindset (I'm from a SLAM background), I organized all my cost functions into _factors_, found in `include/factors`


## `position.cpp`
The first set of unit tests looks at the simplest problems.

### Position1D.AveragePoints
This finds the average of a bunch of 1D samples.  It's a little like opening a piggy bank with a jackhammer, but we gotta start somewhere.  Uses the `Position1DFactor` with analytical (trivial) jacobians.

### Position1D.AveragePointsWithParameterBlock
This is exactly like `Position1D.AveragePoints`, but it adds a (redundant) parameter block.  Just to test how `problem.AddParameterBlock` works.

### Position3D.AveragePoints
This finds the average of a bunch of 1D samples.  Again, overkill, but uses the `Position3DFactor` with analytical jacobians.  I also used this example to explore how Eigen interacts with Ceres.

### Robot1D.SLAM
This performs 1D SLAM (haha!).  There is a 1D robot which takes a 1m step along the real line, 7 times.  Each step, he gets a (noisy) range measurement to 3 landmarks (also on the real line).  It's a pretty trivial problem, but I wanted to explore how to handle multiple factors, and handle measurement variance in my factors.

Uses `Transform1d` - the transform between each step with associated variance and `Range1dFactor` - the range to a landmark and associated variance.

## `attitude.cpp`
The second set of unit tests looks at attitude, and the `LocalParameterization` functionality in ceres.

### Attitude3d.Check*
In writing the `QuatFactor` and `QuatParameterization` classes, I wanted to make sure that the associated `Plus`, `Evaluate` and `ComputeJacobian` functions were correct.  These were pretty straight-forward, except the jacobian of the `QuatFactor`.  That was terrible.  Anyway, these are good for sanity checking the operations being performed by the factor and local parameterization.

### Attitude3d.AverageAttitude
This takes a quaternion and creates 1000 samples normally distributed about this quaternion (using the tangent space to come up with the samples).  Then, I use ceres solver to recover the mean.  Probably the stupidest way to take a mean of a data set, but it exercises the use of a local parameterization, and non-trivial analytic jacobians over my factor.

### Attitude3d.AverageAttitudeAutoDiff
Finds the average attitude of a sample of 1000 attitude measurements using the autodiff functionality for both the localparameterization and cost function.  I also used my templated `Quat` library and let Ceres auto-diff through my library (based on Eigen).

## `pose.cpp`
Next, I figured I could do some pose-graph SLAM in SE3

### Pose.AveragePoseAutoDiff
As before, just to exercise auto differentiation over a tangent space, I used ceres to find the average of a set of 1000 random `Xform` samples. (using my templated `Xform` library to represent members of homogeneous transforms)

### Pose.GraphSLAMAutoDiff
This is the most basic kind of non-trivial SLAM.  We have a bunch of nodes, and edges between nodes.  We also have loop-closures so that the graph is over-constrained and require optimization to find the maximum-likelihood configuration of all the nodes and edges.

I'm using full 6DOF edges and nodes in this graph and the first node is fixed at the origin.

I'm also using auto-differentiated Factors and Local Parameterizations

## `imu.cpp`
Next, I wanted to look into estimating motion of a robot using IMU inputs.  The following examples use a simulated robot that moves at a constant rate along the real line, and but has a noisy IMU with a constant bias.

### IMU.1DRobotSingleWindow
This example looks at a single preintegration window with the origin pose set constant and a direct measurement of the second pose.  Basically, this shows that the unknown IMU bias can be inferred using my `Imu1DFactorAutoDiff` factor.

### IMU.1DRobotLocalization
This example looks at 100 preintegration intervals with with a common constant unknown bias.  The origin is set constant, and the final pose is given a very strong position and velocity measurement.  The IMU bias is inferred using my `IMU1DFactorAutoDiff` factor.

### IMU.1DRobotSLAM
Using the `Range1dFactor` with range measurements to several landmarks also on the real line and the `IMU1dFactor` to perform 1D SLAM.

### IMU.3DRobotSmoothing
In this example, I have a simulate 3D rigid body which moves in space over all 6 degrees of freedom.  I have a simulated IMU with constant accelerometer and rate gyro biases.  I also am using the `XformNodeFactor` to apply measurements of pose directly and therefore infer the biases.

### IMU.3DRobotSLAM
This example is the full SLAM problem - a 3D rigid body moves in space, and has bearing measurements to several landmarks.  Performs SLAM while inferring IMU biases.

### IMU.TransformEstimation
In this example, the IMU has a constant bias on each axis, as well as a constant transform between the IMU and body frame.  The goal of this simulation is to use ceres to estimate this unknown transform.

## `camera.cpp`
Next, I wanted to use the ceres solver to deal with the projection associated with a pinhole camera model.

### Camera.Calibration
This example simulates the calibration of a pinhole camera.  A 3D rigid body gets simulated pixel measurements to known landmarks in the camera FOV. The camera intrinsics are estimated.

### Camera.Transform
This example estimates the transform between a simulated camera, the body frame, camera intrinsics and the IMU frame given pose measurements, pixel measurements, and IMU measurements
