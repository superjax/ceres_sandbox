#pragma once

#include <Eigen/Core>


inline Eigen::MatrixXd calc_jac(std::function<Eigen::MatrixXd(Eigen::MatrixXd)> fun, Eigen::MatrixXd x)
{
  Eigen::MatrixXd y = fun(x);
  int cols = x.rows();
  int rows = y.rows();

  Eigen::MatrixXd I;
  I.resize(cols, cols);
  I.setZero(cols, cols);
  for (int i = 0; i < cols; i++)
    I(i,i) = 1e-8;

  Eigen::MatrixXd JFD;
  JFD.setZero(rows, cols);
  for (int i =0; i < cols; i++)
  {
    Eigen::MatrixXd xp = x + I.col(i);
    Eigen::MatrixXd xm = x - I.col(i);
    Eigen::MatrixXd yp = fun(xp);
    Eigen::MatrixXd ym = fun(xm);
    JFD.col(i) = (yp - ym)/(2*1e-8);
  }
  return JFD;
}
