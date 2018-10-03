#include <ceres/ceres.h>

class Range1dFactor : public ceres::SizedCostFunction<1,1,1>
{
public:
    Range1dFactor(double z, double var) :
        range_(z),
        var_(var)
    {}

    virtual bool Evaluate(const double * const *parameters, double *residuals, double **jacobians) const
    {
        double l = parameters[0][0];
        double x = parameters[1][0];
        residuals[0] = (range_ - (l - x)) / var_;

        if (jacobians)
        {
            if (jacobians[0])
            {
                jacobians[0][0] = -1/var_;
            }
            if (jacobians[1])
            {
                jacobians[1][0] = 1/var_;
            }
        }
        return true;
    }

private:
    double range_;
    double var_;

};

class Range1dFactorVelocity : public ceres::SizedCostFunction<1,1,3>
{
public:
    Range1dFactorVelocity(double z, double var) :
        range_(z),
        var_(var)
    {}

    virtual bool Evaluate(const double * const *parameters, double *residuals, double **jacobians) const
    {
        double l = parameters[0][0];
        double x = parameters[1][0];
//        double v = parameters[1][1];
//        double b = parameters[1][2];
        double neg = 1.0;
        residuals[0] = (range_ - (l - x)) / var_;
        if (residuals[0] < range_ - (l - x))
        {
          neg = -1.0;
          residuals[0] *= 1.0;
        }

        if (jacobians)
        {
            if (jacobians[0])
            {
                jacobians[0][0] = neg * -1.0/var_;
            }
            if (jacobians[1])
            {
                jacobians[1][0] = neg * 1.0/var_;
                jacobians[1][1] = 0;
                jacobians[1][2] = 0;
            }
        }
        return true;
    }

private:
    double range_;
    double var_;

};
