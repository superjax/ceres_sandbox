#pragma once

#include <fstream>
#include <Eigen/Dense>
#include "multirotor_sim/utils.h"


template <typename Scalar>
class Logger
{
public:
    Logger(std::string filename)
    {
        createDirIfNotExist(std::experimental::filesystem::path(filename).parent_path());
        file_.open(filename);
    }

    ~Logger()
    {
        file_.close();
    }
    template <typename... T>
    void log(T... data)
    {
        int dummy[sizeof...(data)] = { (file_.write((char*)&data, sizeof(Scalar)), 1)... };
    }

    template <typename... T>
    void logVectors(T... data)
    {
        int dummy[sizeof...(data)] = { (file_.write((char*)data.data(), sizeof(Scalar)*data.rows()*data.cols()), 1)... };
    }

private:
    std::ofstream file_;
};
