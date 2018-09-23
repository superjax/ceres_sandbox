#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <math.h>
#include <iostream>

using namespace Eigen;

namespace quat {


class Quat
{


private:

public:
  Quat() {}
  Quat(const Vector4d& arr) : arr_(arr) {}
  Quat(const double* data)
  {
      arr_ = Map<const Vector4d>(data);
  }

  inline double* data() { return arr_.data(); }

  Vector4d arr_;
  
  inline double w() const { return arr_(0); }
  inline double x() const { return arr_(1); }
  inline double y() const { return arr_(2); }
  inline double z() const { return arr_(3); }
  inline void setW(double w) { arr_(0) = w; }
  inline void setX(double x) { arr_(1) = x; }
  inline void setY(double y) { arr_(2) = y; }
  inline void setZ(double z) { arr_(3) = z; }
  inline const Vector4d& elements() const { return arr_;}


  Quat operator* (const Quat& q) const { return otimes(q); }
  Quat& operator *= (const Quat& q)
  {
    arr_ <<  w() * q.w() - x() *q.x() - y() * q.y() - z() * q.z(),
             w() * q.x() + x() *q.w() + y() * q.z() - z() * q.y(),
             w() * q.y() - x() *q.z() + y() * q.w() + z() * q.x(),
             w() * q.z() + x() *q.y() - y() * q.x() + z() * q.w();
  }

  Quat& operator= (const Quat& q) { arr_ = q.elements(); }
  Quat& operator= (const Vector4d& in) {arr_ = in; }

  Quat operator+ (const Vector3d& v) { return boxplus(v); }
  Quat& operator+= (const Vector3d& v)
  {
    arr_ = boxplus(v).elements();
  }

  Vector3d operator- (const Quat& q) const {return boxminus(q);}

  static Matrix3d skew(const Vector3d& v)
  {
    static Matrix3d skew_mat;
    skew_mat << 0.0, -v(2), v(1),
                v(2), 0.0, -v(0),
                -v(1), v(0), 0.0;
    return skew_mat;
  }

  static Quat exp(const Vector3d& v)
  {
    double norm_v = v.norm();

    Vector4d q_arr;
    if (norm_v > 1e-4)
    {
      double v_scale = std::sin(norm_v/2.0)/norm_v;
      q_arr << std::cos(norm_v/2.0), v_scale*v(0), v_scale*v(1), v_scale*v(2);
    }
    else
    {
      q_arr << 1.0, v(0)/2.0, v(1)/2.0, v(2)/2.0;
      q_arr /= q_arr.norm();
    }
    return Quat(q_arr);
  }

  static Vector3d log(const Quat& q)
  {
    Vector3d v = q.elements().block<3,1>(1, 0);
    double w = q.elements()(0,0);
    double norm_v = v.norm();

    Vector3d out;
    if (norm_v < 1e-8)
    {
      out.setZero();
    }
    else
    {
      out = 2.0*std::atan2(norm_v, w)*v/norm_v;
    }
    return out;
  }

  static Quat from_R(const Matrix3d& m)
  {
    Vector4d q;
    double tr = m.trace();

    if (tr > 0)
    {
      double S = std::sqrt(tr+1.0) * 2.;
      q << 0.25 * S,
           (m(1,2) - m(2,1)) / S,
           (m(2,0) - m(0,2)) / S,
           (m(0,1) - m(1,0)) / S;
    }
    else if ((m(0,0) > m(1,1)) && (m(0,0) > m(2,2)))
    {
      double S = std::sqrt(1.0 + m(0,0) - m(1,1) - m(2,2)) * 2.;
      q << (m(1,2) - m(2,1)) / S,
           0.25 * S,
           (m(1,0) + m(0,1)) / S,
           (m(2,0) + m(0,2)) / S;
    }
    else if (m(1,1) > m(2,2))
    {
      double S = std::sqrt(1.0 + m(1,1) - m(0,0) - m(2,2)) * 2.;
      q << (m(2,0) - m(0,2)) / S,
           (m(1,0) + m(0,1)) / S,
           0.25 * S,
           (m(2,1) + m(1,2)) / S;
    }
    else
    {
      double S = std::sqrt(1.0 + m(2,2) - m(0,0) - m(1,1)) * 2.;
      q << (m(0,1) - m(1,0)) / S,
           (m(2,0) + m(0,2)) / S,
           (m(2,1) + m(1,2)) / S,
           0.25 * S;
    }
    return Quat(q);
  }

  static Quat from_axis_angle(const Vector3d& axis, const double angle)
  {
    double alpha_2 = angle/2.0;
    double sin_a2 = std::sin(alpha_2);
    Vector4d arr;
    arr << std::cos(alpha_2), axis(0)*sin_a2, axis(1)*sin_a2, axis(2)*sin_a2;
    arr /= arr.norm();
    return Quat(arr);
  }

  static Quat from_euler(const double roll, const double pitch, const double yaw)
  {
    double cp = std::cos(roll/2.0);
    double ct = std::cos(pitch/2.0);
    double cs = std::cos(yaw/2.0);
    double sp = std::sin(roll/2.0);
    double st = std::sin(pitch/2.0);
    double ss = std::sin(yaw/2.0);

    Vector4d arr;
    arr << cp*ct*cs + sp*st*ss,
           sp*ct*cs - cp*st*ss,
           cp*st*cs + sp*ct*ss,
           cp*ct*ss - sp*st*cs;
    return Quat(arr);
  }

  static Quat from_two_unit_vectors(const Vector3d& u, const Vector3d& v)
  {
    Vector4d q_arr;

    double d = u.dot(v);
    if (d < 0.99999999 && d > -0.99999999)
    {
      double invs = 1.0/std::sqrt((2.0*(1.0+d)));
      Vector3d xyz = u.cross(v*invs);
      q_arr(0) = 0.5/invs;
      q_arr.block<3,1>(1,0)=xyz;
      q_arr /= q_arr.norm();
    }
    else if (d < -0.99999999)
    {
      q_arr << 0, 1, 0, 0; // There are an infinite number of solutions here, choose one
    }
    else
    {
      q_arr << 1, 0, 0, 0;
    }
    return Quat(q_arr);
  }

  static Quat Identity()
  {
    Vector4d q_arr;
    q_arr << 1.0, 0, 0, 0;
    return Quat(q_arr);
  }

  static Quat Random()
  {
    Vector4d q_arr;
    q_arr.setRandom();
    q_arr /= q_arr.norm();
    return Quat(q_arr);
  }

  Vector3d euler() const
  {
    Vector3d out;
    out << std::atan2(2.0*(w()*x()+y()*z()), 1.0-2.0*(x()*x() + y()*y())),
        std::asin(2.0*(w()*y() - z()*x())),
        std::atan2(2.0*(w()*z()+x()*y()), 1.0-2.0*(y()*y() + z()*z()));
    return out;
  }
  
  double roll() const
  {
    return std::atan2(2.0*(w()*x()+y()*z()), 1.0-2.0*(x()*x() + y()*y()));
  }
  
  double pitch() const
  {
    return std::asin(2.0*(w()*y() - z()*x()));
  }
  
  double yaw() const
  {
    return std::atan2(2.0*(w()*z()+x()*y()), 1.0-2.0*(y()*y() + z()*z()));
  }

  Vector3d bar() const
  {
    return arr_.segment<3>(1);
  }

  Matrix3d R() const
  {
    double wx = w()*x();
    double wy = w()*y();
    double wz = w()*z();
    double xx = x()*x();
    double xy = x()*y();
    double xz = x()*z();
    double yy = y()*y();
    double yz = y()*z();
    double zz = z()*z();
    Matrix3d out;
    out << 1. - 2.*yy - 2.*zz, 2.*xy + 2.*wz,      2.*xz - 2.*wy,
           2.*xy - 2.*wz,      1. - 2.*xx - 2.*zz, 2.*yz + 2.*wx,
           2.*xz + 2.*wy,      2.*yz - 2.*wx,      1. - 2.*xx - 2.*yy;
    return out;
  }

  Quat copy() const
  {
    Vector4d tmp = arr_;
    return Quat(tmp);
  }

  void normalize()
  {
    arr_ /= arr_.norm();
  }

  Matrix<double, 3, 2> doublerota(const Matrix<double, 3, 2>& v) const
  {
    Matrix<double, 3, 2> out(3, 2);
    Vector3d t;
    for (int i = 0; i < 2; ++i)
    {
      t = 2.0 * v.col(i).cross(bar());
      out.col(i) = v.col(i) - w() * t + t.cross(bar());
    }
    return out;
  }

  Matrix<double, 3, 2> doublerotp(const Matrix<double, 3, 2>& v) const
  {
    Matrix<double, 3, 2> out(3, 2);
    Vector3d t;
    for (int i = 0; i < 2; ++i)
    {
      t = 2.0 * v.col(i).cross(bar());
      out.col(i) = v.col(i) + w() * t + t.cross(bar());
    }
    return out;
  }


  // The same as R.T * v but faster
  Vector3d rota(const Vector3d& v) const
  {
    Vector3d t = 2.0 * v.cross(bar());
    return v - w() * t + t.cross(bar());
  }

  // The same as R * v but faster
  Vector3d rotp(const Vector3d& v) const
  {
    Vector3d t = 2.0 * v.cross(bar());
    return v + w() * t + t.cross(bar());
  }

  Quat& invert()
  {
    arr_.block<3,1>(1,0) *= -1.0;
  }

  Quat inverse() const
  {
    Vector4d tmp = arr_;
    tmp.block<3,1>(1,0) *= -1.0;
    return Quat(tmp);
  }

  Quat otimes(const Quat& q) const
  {
    Vector4d new_arr;
    new_arr <<  w() * q.w() - x() *q.x() - y() * q.y() - z() * q.z(),
                w() * q.x() + x() *q.w() + y() * q.z() - z() * q.y(),
                w() * q.y() - x() *q.z() + y() * q.w() + z() * q.x(),
                w() * q.z() + x() *q.y() - y() * q.x() + z() * q.w();
    return Quat(new_arr);
  }

  Quat boxplus(const Vector3d& delta) const
  {
    return otimes(exp(delta));
  }
  
  Vector3d boxminus(const Quat& q) const
  {
    Quat dq = q.inverse().otimes(*this);
    if (dq.w() < 0.0)
    {
      dq.arr_ *= -1.0;
    }
    return log(dq);
  }
};

inline std::ostream& operator<< (std::ostream& os, const Quat& q)
{
  os << "[ " << q.w() << ", " << q.x() << "i, " << q.y() << "j, " << q.z() << "k]";
  return os;
}

}
