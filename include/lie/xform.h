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

  Xform(const T* data)
  {
    t_ = Map<const Vec3>(data);
    q_ = Map<const Vec4>(data + 3);
  }

  Xform(const Vec7& arr)
    : q_(arr.block(3,0,4,1))
  {
    t_ = arr.block(0,0,3,1);
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

  template<typename T2>
  Matrix<T2,6,1> operator- (const Xform<T2>& X) const
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
    out.block(0,0,3,1) = t_;
    out.block(3,0,4,1) = q_.arr_;
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
    Vec3 u = v.block(0,0,3,1);
    Vec3 omega = v.block(3,0,3,1);
    T th = omega.norm();
    Quat<T> q_exp = Quat<T>::exp(omega);
    if (th > 1e-4)
    {
      Mat3 wx = Quat<T>::skew(omega);
      T B = ((T)1. - cos(th)) / (th * th);
      T C = (th - sin(th)) / (th * th * th);
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
    u.block(3,0,3,1) = omega;
    T th = omega.norm();
    if (th > 1e-16)
    {
      Mat3 wx = Quat<T>::skew(omega);
      T A = sin(th)/th;
      T B = ((T)1. - cos(th)) / (th * th);
      Mat3 V = I_3x3 - (1./2.)*wx + (1./(th*th)) * (1.-(A/(2.*B)))*(wx* wx);
      u.block(0,0,3,1) = V.transpose() * X.t_;
    }
    else
    {
      u.block(0,0,3,1) = X.t_;
    }
    return u;
  }

  Mat6 Adj() const
  {
    Mat6 out;
    Mat3 R = q_.R();
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

  template <typename T2>
  Xform otimes(const Xform<T2>& X2) const
  {
    Xform<T> X;
    X.t_ = t_ + q_.rota(X2.t_);
    X.q_ = q_ * X2.q_;
    return X;
//    return Xform(t_ + q_.rota(X2.t_), q_ * X2.q_);
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

  template<typename T2>
  Matrix<T2,6,1> boxminus(const Xform<T2>& X) const
  {
    return Xform<T2>::log(X.inverse().otimes(*this));
  }

};

template <typename T>
inline std::ostream& operator<< (std::ostream& os, const Xform<T>& X)
{
  os << "t: [ " << X.t_(0,0) << ", " << X.t_(1,0) << ", " << X.t_(2,0) <<
        "] q: [ " << X.q_.w() << ", " << X.q_.x() << "i, " << X.q_.y() << "j, " << X.q_.z() << "k]";
  return os;
}

typedef Xform<double> Xformd;

}
