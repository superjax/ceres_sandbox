#include <gtest/gtest.h>

#include "cam.h"


TEST(Camera, Proj_InvProj)
{
  Camera<double> cam(Vector2d{640, 480});

  cam.focal_len_ << 250, 250;
  cam.cam_center_ << 320, 240;

  Vector3d pt{1.0, 0.5, 2.0};
  double depth = pt.norm();
  Vector2d pix;
  Vector3d pt2;

  cam.proj(pt, pix);
  cam.invProj(pix, depth, pt2);

  EXPECT_NEAR(pt.x(), pt2.x(), 1e-8);
  EXPECT_NEAR(pt.y(), pt2.y(), 1e-8);
  EXPECT_NEAR(pt.z(), pt2.z(), 1e-8);
}
