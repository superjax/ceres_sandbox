#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <math.h>
#include <iostream>

using namespace Eigen;

namespace quat {

template<typename T>
class Quat
{

private:
  typedef Matrix<T,4,1> Vec4;
  typedef Matrix<T,3,1> Vec3;

public:
  Quat() {}
  Quat(const Vec4& arr) : arr_(arr) {}
  Quat(const T* data)
  {
      arr_ = Map<const Vec4>(data);
  }

  inline T* data() { return arr_.data(); }

  Vec4 arr_;
  
  inline T w() const { return arr_(0); }
  inline T x() const { return arr_(1); }
  inline T y() const { return arr_(2); }
  inline T z() const { return arr_(3); }
  inline void setW(T w) { arr_(0) = w; }
  inline void setX(T x) { arr_(1) = x; }
  inline void setY(T y) { arr_(2) = y; }
  inline void setZ(T z) { arr_(3) = z; }
  inline const Vec4& elements() const { return arr_;}


  Quat operator* (const Quat& q) const { return otimes(q); }
  Quat& operator *= (const Quat& q)
  {
    arr_ <<  w() * q.w() - x() *q.x() - y() * q.y() - z() * q.z(),
             w() * q.x() + x() *q.w() + y() * q.z() - z() * q.y(),
             w() * q.y() - x() *q.z() + y() * q.w() + z() * q.x(),
             w() * q.z() + x() *q.y() - y() * q.x() + z() * q.w();
  }

  Quat& operator= (const Quat& q) { arr_ = q.elements(); }
  Quat& operator= (const Vec4& in) {arr_ = in; }

  Quat operator+ (const Vec3& v) { return boxplus(v); }
  Quat& operator+= (const Vec3& v)
  {
    arr_ = boxplus(v).elements();
  }

  template<typename T2>
  Matrix<T2,3,1> operator- (const Quat<T2>& q) const {return boxminus(q);}

  static Matrix<T,3,3> skew(const Vec3& v)
  {
    static Matrix<T,3,3> skew_mat;
    skew_mat << 0.0, -v(2), v(1),
                v(2), 0.0, -v(0),
                -v(1), v(0), 0.0;
    return skew_mat;
  }

  static Quat exp(const Vec3& v)
  {
    T norm_v = v.norm();

    Vec4 q_arr;
    if (norm_v > 1e-4)
    {
      T v_scale = sin(norm_v/2.0)/norm_v;
      q_arr << cos(norm_v/2.0), v_scale*v(0), v_scale*v(1), v_scale*v(2);
    }
    else
    {
      q_arr << (T)1.0, v(0)/2.0, v(1)/2.0, v(2)/2.0;
      q_arr /= q_arr.norm();
    }
    return Quat(q_arr);
  }

  static Vec3 log(const Quat& q)
  {
    Vec3 v = q.arr_.block(1,0,3,1);
    T w = q.w();
    T norm_v = v.norm();

    Vec3 out;
    if (norm_v < (T)1e-8)
    {
      out.setZero();
    }
    else
    {
      out = (T)2.0*atan2(norm_v, w)*v/norm_v;
    }
    return out;
  }

  static Quat from_R(const Matrix<T,3,3>& m)
  {
    Vec4 q;
    T tr = m.trace();

    if (tr > 0)
    {
      T S = sqrt(tr+1.0) * 2.;
      q << 0.25 * S,
           (m(1,2) - m(2,1)) / S,
           (m(2,0) - m(0,2)) / S,
           (m(0,1) - m(1,0)) / S;
    }
    else if ((m(0,0) > m(1,1)) && (m(0,0) > m(2,2)))
    {
      T S = sqrt(1.0 + m(0,0) - m(1,1) - m(2,2)) * 2.;
      q << (m(1,2) - m(2,1)) / S,
           0.25 * S,
           (m(1,0) + m(0,1)) / S,
           (m(2,0) + m(0,2)) / S;
    }
    else if (m(1,1) > m(2,2))
    {
      T S = sqrt(1.0 + m(1,1) - m(0,0) - m(2,2)) * 2.;
      q << (m(2,0) - m(0,2)) / S,
           (m(1,0) + m(0,1)) / S,
           0.25 * S,
           (m(2,1) + m(1,2)) / S;
    }
    else
    {
      T S = sqrt(1.0 + m(2,2) - m(0,0) - m(1,1)) * 2.;
      q << (m(0,1) - m(1,0)) / S,
           (m(2,0) + m(0,2)) / S,
           (m(2,1) + m(1,2)) / S,
           0.25 * S;
    }
    return Quat(q);
  }

  static Quat from_axis_angle(const Vec3& axis, const T angle)
  {
    T alpha_2 = angle/2.0;
    T sin_a2 = sin(alpha_2);
    Vec4 arr;
    arr << cos(alpha_2), axis(0)*sin_a2, axis(1)*sin_a2, axis(2)*sin_a2;
    arr /= arr.norm();
    return Quat(arr);
  }

  static Quat from_euler(const T roll, const T pitch, const T yaw)
  {
    T cp = cos(roll/2.0);
    T ct = cos(pitch/2.0);
    T cs = cos(yaw/2.0);
    T sp = sin(roll/2.0);
    T st = sin(pitch/2.0);
    T ss = sin(yaw/2.0);

    Vec4 arr;
    arr << cp*ct*cs + sp*st*ss,
           sp*ct*cs - cp*st*ss,
           cp*st*cs + sp*ct*ss,
           cp*ct*ss - sp*st*cs;
    return Quat(arr);
  }

  static Quat from_two_unit_vectors(const Vec3& u, const Vec3& v)
  {
    Vec4 q_arr;

    T d = u.dot(v);
    if (d < 0.99999999 && d > -0.99999999)
    {
      T invs = 1.0/sqrt((2.0*(1.0+d)));
      Vec3 xyz = u.cross(v*invs);
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
    Vec4 q_arr;
    q_arr << 1.0, 0, 0, 0;
    return Quat(q_arr);
  }

  static Quat Random()
  {
    Vec4 q_arr;
    q_arr.setRandom();
    q_arr /= q_arr.norm();
    return Quat(q_arr);
  }

  Vec3 euler() const
  {
    Vec3 out;
    out << atan2(2.0*(w()*x()+y()*z()), 1.0-2.0*(x()*x() + y()*y())),
        asin(2.0*(w()*y() - z()*x())),
        atan2(2.0*(w()*z()+x()*y()), 1.0-2.0*(y()*y() + z()*z()));
    return out;
  }
  
  T roll() const
  {
    return atan2(2.0*(w()*x()+y()*z()), 1.0-2.0*(x()*x() + y()*y()));
  }
  
  T pitch() const
  {
    return asin(2.0*(w()*y() - z()*x()));
  }
  
  T yaw() const
  {
    return atan2(2.0*(w()*z()+x()*y()), 1.0-2.0*(y()*y() + z()*z()));
  }

  Vec3 bar() const
  {
    return arr_.segment<3>(1);
  }

  Matrix<T,3,3> R() const
  {
    T wx = w()*x();
    T wy = w()*y();
    T wz = w()*z();
    T xx = x()*x();
    T xy = x()*y();
    T xz = x()*z();
    T yy = y()*y();
    T yz = y()*z();
    T zz = z()*z();
    Matrix<T,3,3> out;
    out << 1. - 2.*yy - 2.*zz, 2.*xy + 2.*wz,      2.*xz - 2.*wy,
           2.*xy - 2.*wz,      1. - 2.*xx - 2.*zz, 2.*yz + 2.*wx,
           2.*xz + 2.*wy,      2.*yz - 2.*wx,      1. - 2.*xx - 2.*yy;
    return out;
  }

  Quat copy() const
  {
    Vec4 tmp = arr_;
    return Quat(tmp);
  }

  void normalize()
  {
    arr_ /= arr_.norm();
  }

  Matrix<T, 3, 2> doublerota(const Matrix<T, 3, 2>& v) const
  {
    Matrix<T, 3, 2> out(3, 2);
    Vec3 t;
    for (int i = 0; i < 2; ++i)
    {
      t = 2.0 * v.col(i).cross(bar());
      out.col(i) = v.col(i) - w() * t + t.cross(bar());
    }
    return out;
  }

  Matrix<T, 3, 2> doublerotp(const Matrix<T, 3, 2>& v) const
  {
    Matrix<T, 3, 2> out(3, 2);
    Vec3 t;
    for (int i = 0; i < 2; ++i)
    {
      t = 2.0 * v.col(i).cross(bar());
      out.col(i) = v.col(i) + w() * t + t.cross(bar());
    }
    return out;
  }


  // The same as R.T * v but faster
  Vec3 rota(const Vec3& v) const
  {
    Vec3 t = 2.0 * v.cross(bar());
    return v - w() * t + t.cross(bar());
  }

  // The same as R * v but faster
  Vec3 rotp(const Vec3& v) const
  {
    Vec3 t = 2.0 * v.cross(bar());
    return v + w() * t + t.cross(bar());
  }

  Quat& invert()
  {
    arr_.block<3,1>(1,0) *= (T)-1.0;
  }

  Quat inverse() const
  {
    Vec4 tmp = arr_;
    tmp(1) *= (T)-1.0;
    tmp(2) *= (T)-1.0;
    tmp(3) *= (T)-1.0;
    return Quat(tmp);
  }

  template <typename T2>
  Quat otimes(const Quat<T2>& q) const
  {
    Vec4 new_arr;
    new_arr <<  w() * q.w() - x() *q.x() - y() * q.y() - z() * q.z(),
                w() * q.x() + x() *q.w() + y() * q.z() - z() * q.y(),
                w() * q.y() - x() *q.z() + y() * q.w() + z() * q.x(),
                w() * q.z() + x() *q.y() - y() * q.x() + z() * q.w();
    return Quat(new_arr);
  }

  Quat boxplus(const Vec3& delta) const
  {
    return otimes(exp(delta));
  }

  template<typename T2>
  Matrix<T2, 3, 1> boxminus(const Quat<T2> &q) const
  {
    Quat<T2> dq = q.inverse().otimes(*this);
    if (dq.w() < 0.0)
    {
      dq.arr_ *= (T2)-1.0;
    }
    return Quat<T2>::log(dq);
  }

};

template<typename T>
inline std::ostream& operator<< (std::ostream& os, const Quat<T>& q)
{
  os << "[ " << q.w() << ", " << q.x() << "i, " << q.y() << "j, " << q.z() << "k]";
  return os;
}

}
