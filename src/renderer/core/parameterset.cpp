#include "renderer/core/parameterset.h"

void ParameterSet::AddInt(const std::string& name, std::vector<int> val)
{
    ASSERT(m_ints.count(name) == 0, "Add an exist attribute " + name);
    m_ints[name] = val;
}

void ParameterSet::AddFloat(const std::string& name, std::vector<Float> val)
{
    ASSERT(m_ints.count(name) == 0, "Add an exist attribute " + name);
    m_floats[name] = val;
}

void ParameterSet::AddString(const std::string& name, std::vector<std::string> val)
{
    ASSERT(m_ints.count(name) == 0, "Add an exist attribute " + name);
    m_strings[name] = val;
}

void ParameterSet::AddPoint(const std::string& name, std::vector<Point3f> val)
{
    ASSERT(m_ints.count(name) == 0, "Add an exist attribute " + name);
    m_points[name] = val;
}

void ParameterSet::AddNormal(const std::string& name, std::vector<Normal3f> val)
{
    ASSERT(m_ints.count(name) == 0, "Add an exist attribute " + name);
    m_normals[name] = val;
}

void ParameterSet::AddRGBSpectrum(const std::string& name, std::vector<Float> val)
{
    ASSERT(m_ints.count(name) == 0, "Add an exist attribute " + name);
    m_rgbSpectrums[name] = val;
}

std::vector<int> ParameterSet::GetInt(const std::string& name) const
{
    if (m_ints.count(name)) {
        return m_ints.find(name)->second;
    }
    else {
        ASSERT(0, "No attribute " + name);
    }
}

std::vector<int> ParameterSet::GetInt(const std::string& name, const std::vector<int> d) const
{
    if (m_ints.count(name)) {
        return m_ints.find(name)->second;
    }
    else {
        return d;
    }
}

std::vector<Float> ParameterSet::GetFloat(const std::string& name) const
{
    if (m_floats.count(name)) {
        return m_floats.find(name)->second;
    }
    else {
        ASSERT(0, "No attribute " + name);
    }
}

std::vector<Float> ParameterSet::GetFloat(const std::string& name, const std::vector<Float> d) const
{
    if (m_floats.count(name)) {
        return m_floats.find(name)->second;
    }
    else {
        return d;
    }
}

std::vector<Point3f> ParameterSet::GetPoint(const std::string& name) const
{
    if (m_points.count(name)) {
        return m_points.find(name)->second;
    }
    else {
        ASSERT(0, "No attribute " + name);
    }
}

std::vector<Point3f> ParameterSet::GetPoint(const std::string& name, const std::vector<Point3f> d) const
{
    if (m_points.count(name)) {
        return m_points.find(name)->second;
    }
    else {
        return d;
    }
}

std::vector<Normal3f> ParameterSet::GetNormal(const std::string& name) const
{
    if (m_normals.count(name)) {
        return m_normals.find(name)->second;
    }
    else {
        ASSERT(0, "No attribute " + name);
    }
}

std::vector<Normal3f> ParameterSet::GetNormal(const std::string& name, const std::vector<Normal3f> d) const
{
    if (m_normals.count(name)) {
        return m_normals.find(name)->second;
    }
    else {
        return d;
    }
}

std::string ParameterSet::GetString(const std::string& name) const
{
    if (m_strings.count(name)) {
        return m_strings.find(name)->second[0];
    }
    else {
        ASSERT(0, "No attribute " + name);
    }
}

std::string ParameterSet::GetString(const std::string& name, const std::string d) const
{
    if (m_strings.count(name)) {
        return m_strings.find(name)->second[0];        
    }
    else {
        return d;
    }
}
