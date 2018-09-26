# Ceres Solver Sandbox

I wanted to learn the [ceres solver](http://ceres-solver.org/).  There are a lot of great tutorials on their wiki, but I felt like I needed some practical experience with some simple problems before I jumped into full-blown SLAM.

So, I broke the problem into little peices and wrote a bunch of simple cases to explore the capabilities of Ceres.  Each section is written as a series of unit tests in the [gtest](https://github.com/google/googletest) framework.

Borrowing off the _factor graph_ mindset (I'm from a SLAM background), I organized all my cost functions into _factors_, found in `include/factors`


## `position.cpp`
The first set of unit tests looks at the simplest problems faced by ceres.

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

### Attitude3d.Check<xxxx>
In writing the `QuatFactor` and `QuatParameterization` classes, I wanted to make sure that the associated `Plus`, `Evaluate` and `ComputeJacobian` functions were correct.  These were pretty straight-forward, except the jacobian of the `QuatFactor`.  That was terrible.  Anyway, these are good for sanity checking the operations being performed by the factor and local parameterization.

### Attitude3d.AverageAttitude
This takes a quaternion and creates 1000 samples normally distributed about this quaternion (using the tangent space to come up with the samples).  Then, I use ceres solver to recover the mean.  Probably the stupidest way to take a mean of a data set, but it exercises the use of a local parameterization, and non-trivial analytic jacobians over my factor.

### Attitude3d.AverageAttitudeAutoGrad
 - Uses the AutoDiff Local Parameterization
 - Uses the AutoDiff Factor

## `pose.cpp`
 