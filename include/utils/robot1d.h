#include <vector>
#include <math.h>
#include <random>
#include <deque>

#include <Eigen/Core>

using namespace Eigen;

class Robot1D
{
public:
  Robot1D(double _ba, double Q, double Td=0.0);

  void add_waypoint(double wp);

  void step(double dt);

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
