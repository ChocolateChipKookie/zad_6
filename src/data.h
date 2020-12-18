//
// Created by kookie on 18. 12. 2020..
//

#ifndef ZAD_6_DATA_H
#define ZAD_6_DATA_H

#include <vector>
#include <array>
#include <iterator>
#include <sstream>
#include <fstream>
#include <cmath>

void createData(const std::string& filepath){
    auto data_function =
        [](double x, double y)
        {
            double x1  = (x - 1.);
            double y2  = (y + 2.);
            double xy5 = 5 * x * y;
            double cosx = std::cos(x/5);
            return (x1*x1 + y2*y2 - xy5 + 3) * cosx * cosx;
        };

    std::pair<int, int> range{-4, 4};

    std::ofstream file(filepath);
    if (!file.is_open()){
        throw std::invalid_argument("File could not be opened!");
    }
    for (int x = range.first; x <= range.second; ++x){
        for (int y = range.first; y <= range.second; ++y){
            file << x << ' ' << y << ' ' << data_function(x, y) << std::endl;
        }
    }
    file.close();
}

std::vector<std::pair<std::vector<double>, double>> loadData(const std::string& filepath){
    std::ifstream file(filepath);
    if (!file.is_open()){
        throw std::invalid_argument("File could not be opened!");
    }
    std::vector<std::pair<std::vector<double>, double>> res;
    // Predefined number of data instances 9x9
    res.reserve(81);

    std::string line;
    while (std::getline(file, line)){
        std::stringstream ss(line);
        std::istream_iterator<double> begin{ss}, end{};
        std::vector<double> tmp{begin, end};
        res.push_back({{tmp.begin(), tmp.end()-1}, tmp.back()});
    }
    return res;
}

#endif //ZAD_6_DATA_H
