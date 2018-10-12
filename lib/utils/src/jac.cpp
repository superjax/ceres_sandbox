#include <Eigen/Core>
#include "utils/jac.h"

Eigen::MatrixXd calc_jac(std::function<Eigen::MatrixXd(const Eigen::MatrixXd&)> fun, const Eigen::MatrixXd& x,
                         std::function<Eigen::MatrixXd(const Eigen::MatrixXd&, const Eigen::MatrixXd&)>f_boxminus,
                         std::function<Eigen::MatrixXd(const Eigen::MatrixXd&, const Eigen::MatrixXd&)>f_boxplus,
                         std::function<Eigen::MatrixXd(const Eigen::MatrixXd&, const Eigen::MatrixXd&)>f_boxminus2,
                         std::function<Eigen::MatrixXd(const Eigen::MatrixXd&, const Eigen::MatrixXd&)>f_boxplus2)
{
  if (f_boxminus == nullptr)
  {
      f_boxminus = [](const Eigen::MatrixXd& x1, const Eigen::MatrixXd& x2)
      {
          return x1 - x2;
      };
  }
  if (f_boxminus2 == nullptr)
  {
      f_boxminus2 = [](const Eigen::MatrixXd& x1, const Eigen::MatrixXd& x2)
      {
          return x1 - x2;
      };
  }

  if (f_boxplus == nullptr)
  {
      f_boxplus = [](const Eigen::MatrixXd& x1, const Eigen::MatrixXd& dx)
      {
          return x1 + dx;
      };
  }
  if (f_boxplus2 == nullptr)
  {
      f_boxplus2 = [](const Eigen::MatrixXd& x1, const Eigen::MatrixXd& dx)
      {
          return x1 + dx;
      };
  }

  Eigen::MatrixXd y = fun(x);
  Eigen::MatrixXd dy = f_boxminus2(y,y);
  int rows = dy.rows();

  Eigen::MatrixXd dx = f_boxminus(x, x);
  int cols = dx.rows();

  Eigen::MatrixXd I;
  I.resize(cols, cols);
  I.setZero(cols, cols);
  for (int i = 0; i < cols; i++)
    I(i,i) = 1e-8;

  Eigen::MatrixXd JFD;
  JFD.setZero(rows, cols);
  for (int i =0; i < cols; i++)
  {
    Eigen::MatrixXd xp = f_boxplus(x, I.col(i));
    Eigen::MatrixXd xm = f_boxplus(x, -1.0*I.col(i));

    Eigen::MatrixXd yp = fun(xp);
    Eigen::MatrixXd ym = fun(xm);
    Eigen::MatrixXd dy = f_boxminus2(yp,ym);

    JFD.col(i) = dy/(2*1e-8);
  }
  return JFD;
}
