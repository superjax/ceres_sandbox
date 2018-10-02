#include <vector>
#include <math.h>
#include <random>

class Robot1D
{
public:
    Robot1D(double _ba, double var) : normal_(0.0, sqrt(var))
    {
        x_ = 0;
        a_ = 0;
        v_ = 0;
        i_ = 0;
        t_ = 0;
        prev_x_ = NAN;
        kp_ = 0.3;
        kd_ = 0.003;
        ba_ = _ba;

        xhat_ = x_;
        vhat_ = v_;
        ahat_ = a_;
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

        // propagate estimates
        ahat_ = a_ + ba_;
    }

    double xhat_;
    double vhat_;
    double ahat_;
    double t_;
    std::vector<double> waypoints_;

    double ba_;
    double x_;
    double v_;
    double a_;
    double kp_;
    double kd_;
    double prev_x_;
    int i_;

    std::default_random_engine gen_;
    std::normal_distribution<double> normal_;


};
