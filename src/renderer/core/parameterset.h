#pragma once
#ifndef __PARAMETERSET_H
#define __PARAMETERSET_H

#include "renderer/core/fwd.h"
#include "renderer/core/geometry.h"

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

    std::vector<int> GetInt(const std::string& name) const;
    std::vector<int> GetInt(const std::string& name, const std::vector<int> d) const;
    std::vector<Float> GetFloat(const std::string& name) const;
    std::vector<Float> GetFloat(const std::string& name, const std::vector<Float> d) const;
    std::vector<Point3f> GetPoint(const std::string& name) const;
    std::vector<Point3f> GetPoint(const std::string& name, const std::vector<Point3f> d) const;
    std::vector<Normal3f> GetNormal(const std::string& name) const;
    std::vector<Normal3f> GetNormal(const std::string& name, const std::vector<Normal3f> d) const;
    std::string GetString(const std::string& name) const;
    std::string GetString(const std::string& name, const std::string d) const;


    std::map<std::string, std::vector<int>> m_ints;
    std::map<std::string, std::vector<Float>> m_floats;
    std::map<std::string, std::vector<std::string>> m_strings;
    std::map<std::string, std::vector<Point3f>> m_points;
    std::map<std::string, std::vector<Normal3f>> m_normals;
    std::map<std::string, std::vector<Float>> m_rgbSpectrums;
};

#endif // !__PARAMETERSET_H
