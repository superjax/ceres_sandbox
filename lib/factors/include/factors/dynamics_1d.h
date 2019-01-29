#include <ceres/ceres.h>


class Dyn1DFunctor
{
public:
  Dyn1DFunctor(double dt)
  {
    dt_ = dt;
  }

  template <typename T>
  bool operator() (const T x0[2], const T x1[2], const T u0[1], const T u1[1], T res[2]) const
  {
    // trapezoidal integration
    res[0] = x1[0] - (x0[0] + (T)0.5 * (T)dt_ * (x1[1] + x0[1]));
    res[1] = x1[1] - (x0[1] + (T)0.5 * (T)dt_ * (u1[0] + u0[0]));

    res[0] *= (T) 1e6;
    res[1] *= (T) 1e6;
    return true;
  }
  double dt_;
};
typedef ceres::AutoDiffCostFunction<Dyn1DFunctor, 2, 2, 2, 1, 1> Dyn1dFactorAD;


class PosVel1DFunctor
{
public:
  PosVel1DFunctor(double x, double v)
  {
    x_ = x;
    v_ = v;
  }
  template <typename T>
  bool operator() (const T* x, T* res) const
  {
    res[0] = x[0] - (T)x_;
    res[1] = x[1] - (T)v_;

    res[0] *= (T) 1e6;
    res[1] *= (T) 1e6;
    return true;
  }

  double x_;
  double v_;
};
typedef ceres::AutoDiffCostFunction<PosVel1DFunctor, 2, 2> PosVel1DFactorAD;


class Input1DFunctor
{
public:
  Input1DFunctor(){}

  template <typename T>
  bool operator() (const T* u, T* res) const
  {
    res[0] = u[0];
    return true;
  }
};
typedef ceres::AutoDiffCostFunction<Input1DFunctor, 1, 1> Input1DFactorAD;
