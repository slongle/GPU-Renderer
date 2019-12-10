#pragma once
#ifndef __PARAMETERSET_H
#define __PARAMETERSET_H

#include "renderer/core/fwd.h"

#include <map>
#include <string>
#include <vector>

class ParameterSet {
public:

    void AddInt(const std::string& name, std::vector<int> val);
    void AddFloat(const std::string& name, std::vector<Float> val);
    void AddString(const std::string& name, std::vector<std::string> val);
    void AddPoint(const std::string& name, std::vector<Point3f> val);
    void AddNormal(const std::string& name, std::vector<Normal3f> val);
    void AddRGBSpectrum(const std::string& name, std::vector<Float> val);


    std::map<std::string, std::vector<int>> m_ints;
    std::map<std::string, std::vector<Float>> m_floats;
    std::map<std::string, std::vector<std::string>> m_strings;
    std::map<std::string, std::vector<Point3f>> m_points;
    std::map<std::string, std::vector<Normal3f>> m_normals;
    std::map<std::string, std::vector<Float>> m_rgbSpectrums;
};

#endif // !__PARAMETERSET_H
