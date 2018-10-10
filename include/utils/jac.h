#include <Eigen/Core>

Eigen::MatrixXd calc_jac(std::function<Eigen::MatrixXd(Eigen::MatrixXd)> fun, Eigen::MatrixXd x,
                         std::function<Eigen::MatrixXd(const Eigen::MatrixXd&, const Eigen::MatrixXd&)>f_boxminus=nullptr,
                         std::function<Eigen::MatrixXd(const Eigen::MatrixXd&, const Eigen::MatrixXd&)>f_boxplus=nullptr,
                         std::function<Eigen::MatrixXd(const Eigen::MatrixXd &, const Eigen::MatrixXd &)> f_boxminus2=nullptr,
                         std::function<Eigen::MatrixXd(const Eigen::MatrixXd &, const Eigen::MatrixXd &)> f_boxplus2=nullptr);
