#include <Eigen/Core>

Eigen::MatrixXd calc_jac(std::function<Eigen::MatrixXd(const Eigen::MatrixXd &)> fun, const Eigen::MatrixXd &x,
                         std::function<Eigen::MatrixXd(const Eigen::MatrixXd&, const Eigen::MatrixXd&)>x_boxminus=nullptr,
                         std::function<Eigen::MatrixXd(const Eigen::MatrixXd&, const Eigen::MatrixXd&)>x_boxplus=nullptr,
                         std::function<Eigen::MatrixXd(const Eigen::MatrixXd &, const Eigen::MatrixXd &)> y_boxminus=nullptr, double step_size=1e-8);
