#include <gtest/gtest.h>

#define ASSERT_MAT_EQ(v1, v2) \
{ \
    ASSERT_EQ((v1).rows(), (v2).rows()); \
    ASSERT_EQ((v1).cols(), (v2).cols()); \
    for (int row = 0; row < (v1).rows(); row++) {\
        for (int col = 0; col < (v2).cols(); col++) {\
            ASSERT_FLOAT_EQ((v1)(row, col), (v2)(row,col));\
        }\
    }\
}

#define ASSERT_MAT_NEAR(v1, v2, tol) \
{ \
    ASSERT_EQ((v1).rows(), (v2).rows()); \
    ASSERT_EQ((v1).cols(), (v2).cols()); \
    for (int row = 0; row < (v1).rows(); row++) {\
        for (int col = 0; col < (v2).cols(); col++) {\
            ASSERT_NEAR((v1)(row, col), (v2)(row,col), (tol));\
        }\
    }\
}

#define EXPECT_MAT_NEAR(v1, v2, tol) \
{ \
    EXPECT_EQ((v1).rows(), (v2).rows()); \
    EXPECT_EQ((v1).cols(), (v2).cols()); \
    for (int row = 0; row < (v1).rows(); row++) {\
        for (int col = 0; col < (v2).cols(); col++) {\
            EXPECT_NEAR((v1)(row, col), (v2)(row,col), (tol));\
        }\
    }\
}

#define ASSERT_XFORM_NEAR(x1, x2, tol) \
{ \
    ASSERT_NEAR((x1).t()(0), (x2).t()(0), tol);\
    ASSERT_NEAR((x1).t()(1), (x2).t()(1), tol);\
    ASSERT_NEAR((x1).t()(2), (x2).t()(2), tol);\
    ASSERT_NEAR((x1).q().w(), (x2).q().w(), tol);\
    ASSERT_NEAR((x1).q().x(), (x2).q().x(), tol);\
    ASSERT_NEAR((x1).q().y(), (x2).q().y(), tol);\
    ASSERT_NEAR((x1).q().z(), (x2).q().z(), tol);\
}

#define ASSERT_QUAT_NEAR(q1, q2, tol) \
do { \
    Vector3d qt = (q1) - (q2);\
    ASSERT_LE(std::abs(qt(0)), tol);\
    ASSERT_LE(std::abs(qt(1)), tol);\
    ASSERT_LE(std::abs(qt(2)), tol);\
} while(0)

#define EXPECT_QUAT_NEAR(q1, q2, tol) \
do { \
    Vector3d qt = (q1) - (q2);\
    EXPECT_LE(std::abs(qt(0)), tol);\
    EXPECT_LE(std::abs(qt(1)), tol);\
    EXPECT_LE(std::abs(qt(2)), tol);\
} while(0)
