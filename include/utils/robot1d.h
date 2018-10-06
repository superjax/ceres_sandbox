#include <vector>
#include <math.h>
#include <random>
#include <deque>

#include <Eigen/Core>

using namespace Eigen;

class Robot1D
{
public:
  Robot1D(double _ba, double Q, double Td=0.0) :
    normal_(0.0, 1.0)
  {
    x_ = 0;
    a_ = 0;
    v_ = 0;
    i_ = 0;
    t_ = 0;
    prev_x_ = NAN;
    kp_ = 0.3;
    kd_ = 0.003;
    b_ = _ba;
    Td_ = Td;

    a_stdev_ = Q;
    b_stdev_ = 0.0;

    xhat_ = x_;
    vhat_ = v_;
    ahat_ = a_;

    // Create a history
    history_t history {x_, v_, t_, xhat_, vhat_};
    hist_.push_back(history);
  }

  void add_waypoint(double wp)
  {
    waypoints_.push_back(wp);
  }

  void step(double dt)
  {
    if (std::abs(x_ - waypoints_[i_]) < 1e-2)
      i_ = (i_ + 1) % waypoints_.size();

    // propagate dynamics
    double e = waypoints_[i_] - x_;
    if (!std::isfinite(prev_x_))
      prev_x_ = x_;
    a_ = kp_*e + kd_ * (prev_x_ - x_)/dt;
    x_ += v_ * dt;
    v_ += a_ * dt;
    t_ += dt;
    b_ += normal_(gen_)*b_stdev_*dt;

    // propagate estimates
    ahat_ = a_ + normal_(gen_)*a_stdev_ - b_;

    // Save history
    history_t history {x_, v_, t_, xhat_, vhat_};
    hist_.push_back(history);
    while (t_ - hist_.front().t > Td_ && hist_.size() > 0)
    {
      hist_.pop_front();
    }
  }

  double xhat_;
  double vhat_;
  double ahat_;
  double t_;
  std::vector<double> waypoints_;

  typedef struct
  {
    double x;
    double v;
    double t;
    double xhat;
    double vhat;
  } history_t;
  std::deque<history_t> hist_;

  double Td_;
  double b_;
  double x_;
  double v_;
  double a_;
  double kp_;
  double kd_;
  double prev_x_;
  int i_;

  double a_stdev_;
  double b_stdev_;

  std::default_random_engine gen_;
  std::normal_distribution<double> normal_;


};
