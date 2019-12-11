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
