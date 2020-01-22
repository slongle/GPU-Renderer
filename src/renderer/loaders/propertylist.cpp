#include "propertylist.h"

#define ADD_PROPERTYLIST_FUNCTIONS(Type, Typename, XMLName) \
    void PropertyList::set##Typename(const std::string &name, const Type &value) { \
        Property &property = list[name]; \
        property.type = Property::EP##Typename; \
        property.value.XMLName##_value = value; \
    } \
    \
    Type PropertyList::get##Typename(const std::string &name) const { \
        assert(list.find(name) != list.end(), "Not Found The Value Named" + name); \
        auto it = list.find(name); \
        assert(it->second.type == Property::EP##Typename, "Wrong Value's Type"); \
        return it->second.value.XMLName##_value; \
    } \
    \
    Type PropertyList::get##Typename(const std::string &name, const Type &defaultValue) const { \
        if (list.find(name) == list.end()) \
        return defaultValue; \
	    auto it = list.find(name); \
	    assert(it->second.type == Property::EP##Typename, "Wrong Value's Type"); \
	    return it->second.value.XMLName##_value; \
	} \
    \
    bool PropertyList::find##Typename(const std::string &name) const { \
        if (list.find(name) != list.end()){ \
            auto it = list.find(name); \
            assert(it->second.type == Property::EP##Typename, "Wrong Value's Type"); \
            return true; \
        } else { \
            return false; \
        } \
    }



ADD_PROPERTYLIST_FUNCTIONS(bool, Boolean, boolean)
ADD_PROPERTYLIST_FUNCTIONS(int, Integer, integer)
ADD_PROPERTYLIST_FUNCTIONS(float, Float, float)
ADD_PROPERTYLIST_FUNCTIONS(std::string, String, string)
ADD_PROPERTYLIST_FUNCTIONS(float3, Vector, vector)
ADD_PROPERTYLIST_FUNCTIONS(Spectrum, Color, color)
ADD_PROPERTYLIST_FUNCTIONS(Transform, Transform, transform)