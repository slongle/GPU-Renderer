#pragma once

#include "renderer/fwd.h"
#include "renderer/spectrum.h"
#include "renderer/transform.h"

struct Property {
    Property() {}

    enum {
        EPBoolean,
        EPInteger,
        EPFloat,
        EPString,
        EPVector,
        EPColor,
        EPTransform
    } type;

    struct Value {
        Value() {}

        bool        boolean_value;
        int         integer_value;
        float       float_value;
        std::string string_value;
        float3      vector_value;
        Spectrum    color_value;
        Transform   transform_value;
    } value;
};

class PropertyList {
public:
    void setBoolean(const std::string& name, const bool& value);
    bool getBoolean(const std::string& name) const;
    bool getBoolean(const std::string& name, const bool& defaultValue) const;
    bool findBoolean(const std::string& name) const;

    void setInteger(const std::string& name, const int& value);
    int  getInteger(const std::string& name) const;
    int  getInteger(const std::string& name, const int& defaultValue) const;
    bool findInteger(const std::string& name) const;

    void  setFloat(const std::string& name, const float& value);
    float getFloat(const std::string& name) const;
    float getFloat(const std::string& name, const float& defaultValue) const;
    bool findFloat(const std::string& name) const;

    void setString(const std::string& name, const std::string& value);
    std::string getString(const std::string& name) const;
    std::string getString(const std::string& name, const std::string& defaultValue) const;
    bool findString(const std::string& name) const;

    void setVector(const std::string& name, const float3& value);
    float3 getVector(const std::string& name) const;
    float3 getVector(const std::string& name, const float3& defaultValue) const;
    bool findVector(const std::string& name) const;

    void setColor(const std::string& name, const Spectrum& value);
    Spectrum getColor(const std::string& name) const;
    Spectrum getColor(const std::string& name, const Spectrum& defaultValue) const;
    bool findColor(const std::string& name) const;

    void setTransform(const std::string& name, const Transform& value);
    Transform getTransform(const std::string& name) const;
    Transform getTransform(const std::string& name, const Transform& defaultValue) const;
    bool findTransform(const std::string& name) const;

private:
    std::map<std::string, Property> list;

};