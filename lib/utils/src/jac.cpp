#include <Eigen/Core>
#include "utils/jac.h"

Eigen::MatrixXd calc_jac(std::function<Eigen::MatrixXd(const Eigen::MatrixXd&)> fun, const Eigen::MatrixXd& x,
                         std::function<Eigen::MatrixXd(const Eigen::MatrixXd&, const Eigen::MatrixXd&)>x_boxminus,
                         std::function<Eigen::MatrixXd(const Eigen::MatrixXd&, const Eigen::MatrixXd&)>x_boxplus,
                         std::function<Eigen::MatrixXd(const Eigen::MatrixXd&, const Eigen::MatrixXd&)>y_boxminus,
                         double step_size)
{
  if (x_boxminus == nullptr)
  {
      x_boxminus = [](const Eigen::MatrixXd& x1, const Eigen::MatrixXd& x2)
      {
          return x1 - x2;
      };
  }
  if (y_boxminus == nullptr)
  {
      y_boxminus = [](const Eigen::MatrixXd& x1, const Eigen::MatrixXd& x2)
      {
          return x1 - x2;
      };
  }

  if (x_boxplus == nullptr)
  {
      x_boxplus = [](const Eigen::MatrixXd& x1, const Eigen::MatrixXd& dx)
      {
          return x1 + dx;
      };
  }

  Eigen::MatrixXd y = fun(x);
  Eigen::MatrixXd dy = y_boxminus(y,y);
  int rows = dy.rows();

  Eigen::MatrixXd dx = x_boxminus(x, x);
  int cols = dx.rows();

  Eigen::MatrixXd I;
  I.resize(cols, cols);
  I.setZero(cols, cols);
  for (int i = 0; i < cols; i++)
    I(i,i) = step_size;

  Eigen::MatrixXd JFD;
  JFD.setZero(rows, cols);
  for (int i =0; i < cols; i++)
  {
    Eigen::MatrixXd xp = x_boxplus(x, I.col(i));
    Eigen::MatrixXd xm = x_boxplus(x, -1.0*I.col(i));

    Eigen::MatrixXd yp = fun(xp);
    Eigen::MatrixXd ym = fun(xm);
    Eigen::MatrixXd dy = y_boxminus(yp,ym);

    JFD.col(i) = dy/(2*step_size);
  }
  return JFD;
}
