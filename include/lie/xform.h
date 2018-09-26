#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <math.h>
#include <iostream>

#include "lie/quat.h"
#include "lie/math_helper.h"

using namespace Eigen;
using namespace quat;

namespace xform
{

template <typename T>
class Xform
{
private:

  typedef Matrix<T, 2, 1> Vec2;
  typedef Matrix<T, 3, 1> Vec3;
  typedef Matrix<T, 4, 1> Vec4;
  typedef Matrix<T, 5, 1> Vec5;
  typedef Matrix<T, 6, 1> Vec6;
  typedef Matrix<T, 7, 1> Vec7;

  typedef Matrix<T, 3, 3> Mat3;
  typedef Matrix<T, 4, 4> Mat4;
  typedef Matrix<T, 6, 6> Mat6;

public:
  Vec3 t_;
  Quat<T> q_;

  Xform(){}

  Xform(const double* data)
  {
    t_ = Map<const Vec3>(data);
    q_ = Map<const Vec4>(data + 3);
  }

  Xform(const Vec7& arr)
    : q_(arr.segment<4>(3))
  {
    t_ = arr.segment<3>(0);
  }

  Xform(const Vec3& t, const Quat<T>& q)
  {
    t_ = t;
    q_ = q;
  }

  Xform(const Vec3& t, const Mat3& R)
  {
    q_ = Quat<T>::from_R(R);
    t_ = t;
  }

  Xform(const Mat4& X)
  {
    q_ = Quat<T>::from_R(X.block<3,3>(0,0));
    t_ = X.block<3,1>(0, 3);
  }

  inline Vec3& t() { return t_;}
  inline Quat<T>& q() { return q_;}
  inline void setq(const Quat<T>& q) {q_ = q;}
  inline void sett(const Vec3&t) {t_ = t;}

  Xform operator* (const Xform& X) const {return otimes(X);}
  Xform& operator*= (const Xform& X)
  {
    t_ = t_ + q_.rotp(X.t_);
    q_ = q_ * X.q_;
  }
  Xform& operator=(const Xform& X) {t_ = X.t_; q_ = X.q_;}
  Xform& operator=(const Vec7& v) {
    t_ = v.segment<3>(0);
    q_ = Quat<T>(v.segment<4>(3));
  }

  Xform operator+ (const Vec6& v)
  {
    return boxplus(v);
  }

  Vec6 operator- (const Xform& X)
  {
    return boxminus(X);
  }

  Xform& operator+=(const Vec6& v)
  {
    *this = boxplus(v);
  }

  Vec7 elements() const
  {
    Vec7 out;
    out.segment<3>(0) = t_;
    out.segment<4>(3) = q_.arr_;
    return out;
  }

  Mat4 Mat() const
  {
    Mat4 out;
    out.block<3,3>(0,0) = q_.R();
    out.block<3,1>(0,3) = t_;
    out.block<1,3>(3,0) = Matrix<T,1,3>::Zero();
    out(3,3) = 1.0;
  }

  static Xform Identity()
  {
    Xform out;
    out.t_.setZero();
    out.q_ = Quat<T>::Identity();
    return out;
  }

  static Xform Random()
  {
    Xform out;
    out.t_.setRandom();
    out.q_ = Quat<T>::Random();
    return out;
  }

  static Xform exp(const Vec6& v)
  {
    Vec3 u = v.segment<3>(0);
    Vec3 omega = v.segment<3>(3);
    double th = omega.norm();
    Quat<T> q_exp = Quat<T>::exp(omega);
    if (th > 1e-4)
    {
      Matrix3d wx = Quat<T>::skew(omega);
      double B = (1. - std::cos(th)) / (th * th);
      double C = (th - std::sin(th)) / (th * th * th);
      return Xform((I_3x3 + B*wx + C*wx*wx).transpose() * u, q_exp);
    }
    else
    {
      return Xform(u, q_exp);
    }
  }

  static Vec6 log(const Xform& X)
  {
    Vec6 u;
    Vec3 omega = Quat<T>::log(X.q_);
    u.segment<3>(3) = omega;
    double th = omega.norm();
    if (th > 1e-16)
    {
      Matrix3d wx = Quat<T>::skew(omega);
      double A = std::sin(th)/th;
      double B = (1. - std::cos(th)) / (th * th);
      Matrix3d V = I_3x3 - (1./2.)*wx + (1./(th*th)) * (1.-(A/(2.*B)))*(wx* wx);
      u.segment<3>(0) = V.transpose() * X.t_;
    }
    else
    {
      u.segment<3>(0) = X.t_;
    }
    return u;
  }

  Mat6 Adj() const
  {
    Mat6 out;
    Matrix3d R = q_.R();
    out.block<3,3>(0,0) = R;
    out.block<3,3>(0,3) = Quat<T>::skew(t_)*R;
    out.block<3,3>(3,3) = R;
    out.block<3,3>(3,0) = Mat3::Zero();
    return out;
  }

  Xform inverse() const{
    Xform out(-q_.rotp(t_), q_.inverse());
    return out;
  }

  Xform otimes(const Xform& T2) const
  {
    return Xform(t_ + q_.rota(T2.t_), q_ * T2.q_);
  }

  Vec3 transforma(const Vec3& v) const
  {
    return q_.rota(v) + t_;
  }

  Vec3 transformp(const Vec3& v) const
  {
    return q_.rotp(v - t_);
  }

  Xform& invert()
  {
    t_ = -q_.rotp(t_);
    q_.invert();
  }

  Xform boxplus(const Vec6& delta) const
  {
    return otimes(Xform::exp(delta));
  }

  Vec6 boxminus(const Xform& X) const
  {
    return Xform::log(X.inverse().otimes(*this));
  }

};

template <typename T>
inline std::ostream& operator<< (std::ostream& os, const Xform<T>& X)
{
  os << "t: [ " << X.t_(0,0) << ", " << X.t_(1,0) << ", " << X.t_(2,0) <<
        "] q: [ " << X.q_.w() << ", " << X.q_.x() << "i, " << X.q_.y() << "j, " << X.q_.z() << "k]";
  return os;
}

}
