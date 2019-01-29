#include <ceres/ceres.h>

class Transform1DFactor : public ceres::SizedCostFunction<1,1,1>
{
public:
    Transform1DFactor(double z, double var) :
        transform_(z),
        var_(var)
    {}

    virtual bool Evaluate(const double * const *parameters, double *residuals, double **jacobians) const
    {
        double xi = parameters[0][0];
        double xj = parameters[1][0];
        residuals[0] = (transform_ - (xj - xi))/var_;

        if (jacobians)
        {
            if (jacobians[0])
            {
                jacobians[0][0] = 1.0/var_;
            }
            if (jacobians[1])
            {
                jacobians[1][0] = -1.0/var_;
            }
        }
        return true;
    }

private:
    double transform_;
    double var_;
};
