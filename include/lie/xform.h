#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <math.h>
#include <iostream>

#include "quat.h"
#include "math_helper.h"

using namespace Eigen;
using namespace quat;

typedef Matrix<double, 7, 1> Vector7d;
typedef Matrix<double, 6, 1> Vector6d;
typedef Matrix<double, 6, 6> Matrix6d;
typedef Matrix<double, 4, 4> Matrix4d;

namespace xform
{

class Xform
{
private:
public:
  Vector3d t_;
  Quat q_;

  Xform(){}

  Xform(const Vector7d& arr)
  {
    t_ = arr.segment<3>(0);
    q_ = Quat(arr.segment<4>(3));
  }

  Xform(const Vector3d& t, const Quat& q)
  {
    t_ = t;
    q_ = q;
  }

  Xform(const Vector3d& t, const Matrix3d& R)
  {
    q_ = Quat::from_R(R);
    t_ = t;
  }

  Xform(const Matrix4d& T)
  {
    q_ = Quat::from_R(T.block<3,3>(0,0));
    t_ = T.block<3,1>(0, 3);
  }

  inline Vector3d& t() { return t_;}
  inline Quat& q() { return q_;}
  inline void setq(const Quat& q) {q_ = q;}
  inline void sett(const Vector3d&t) {t_ = t;}

  Xform operator* (const Xform& T) const {return otimes(T);}
  Xform& operator*= (const Xform& T)
  {
    t_ = t_ + q_.rotp(T.t_);
    q_ = q_ * T.q_;
  }
  Xform& operator=(const Xform& T) {t_ = T.t_; q_ = T.q_;}
  Xform& operator=(const Vector7d& v) {
    t_ = v.segment<3>(0);
    q_ = Quat(v.segment<4>(3));
  }

  Xform operator+ (const Vector6d& v)
  {
    return boxplus(v);
  }

  Vector6d operator- (const Xform& T)
  {
    return boxminus(T);
  }

  Xform& operator+=(const Vector6d& v)
  {
    *this = boxplus(v);
  }

  Vector7d elements() const
  {
    Vector7d out;
    out.segment<3>(0) = t_;
    out.segment<4>(3) = q_.arr_;
    return out;
  }

  Matrix4d T() const
  {
    Matrix4d out;
    out.block<3,3>(0,0) = q_.R();
    out.block<3,1>(0,3) = t_;
    out.block<1,3>(3,0).setZero();
    out(3,3) = 1.0;
  }

  static Xform Identity()
  {
    Xform out;
    out.t_.setZero();
    out.q_ = Quat::Identity();
    return out;
  }

  static Xform Random()
  {
    Xform out;
    out.t_.setRandom();
    out.q_ = Quat::Random();
    return out;
  }

  static Xform exp(const Vector6d& v)
  {
    Vector3d u = v.segment<3>(0);
    Vector3d omega = v.segment<3>(3);
    double th = omega.norm();
    Quat q_exp = Quat::exp(omega);
    if (th > 1e-4)
    {
      Matrix3d wx = Quat::skew(omega);
      double B = (1. - std::cos(th)) / (th * th);
      double C = (th - std::sin(th)) / (th * th * th);
      return Xform((I_3x3 + B*wx + C*wx*wx).transpose() * u, q_exp);
    }
    else
    {
      return Xform(u, q_exp);
    }
  }

  static Vector6d log(const Xform& T)
  {
    Vector6d u;
    Vector3d omega = Quat::log(T.q_);
    u.segment<3>(3) = omega;
    double th = omega.norm();
    if (th > 1e-16)
    {
      Matrix3d wx = Quat::skew(omega);
      double A = std::sin(th)/th;
      double B = (1. - std::cos(th)) / (th * th);
      Matrix3d V = I_3x3 - (1./2.)*wx + (1./(th*th)) * (1.-(A/(2.*B)))*(wx* wx);
      u.segment<3>(0) = V.transpose() * T.t_;
    }
    else
    {
      u.segment<3>(0) = T.t_;
    }
    return u;
  }

  Matrix6d Adj() const
  {
    Matrix6d out;
    Matrix3d R = q_.R();
    out.block<3,3>(0,0) = R;
    out.block<3,3>(0,3) = Quat::skew(t_)*R;
    out.block<3,3>(3,3) = R;
    out.block<3,3>(3,0).setZero();
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

  Vector3d transforma(const Vector3d& v) const
  {
    return q_.rota(v) + t_;
  }

  Vector3d transformp(const Vector3d& v) const
  {
    return q_.rotp(v - t_);
  }

  Xform& invert()
  {
    t_ = -q_.rotp(t_);
    q_.invert();
  }

  Xform boxplus(const Vector6d& delta) const
  {
    return otimes(Xform::exp(delta));
  }

  Vector6d boxminus(const Xform& T) const
  {
    return Xform::log(T.inverse().otimes(*this));
  }

};

inline std::ostream& operator<< (std::ostream& os, const Xform& T)
{
  os << "t: [ " << T.t_(0,0) << ", " << T.t_(1,0) << ", " << T.t_(2,0) <<
        "] q: [ " << T.q_.w() << ", " << T.q_.x() << "i, " << T.q_.y() << "j, " << T.q_.z() << "k]";
  return os;
}

}
