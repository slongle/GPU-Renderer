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

void PropertyList::setTexture(
    const std::string& name, 
    std::shared_ptr<Texture> value)
{
    Property& property = list[name];
    property.type = Property::EPTexture;
    property.value.texture_value = value;
}


std::shared_ptr<Texture>
PropertyList::getTexture(
    const std::string& name) const
{
    assert(list.find(name) != list.end(), "Not Found The Value Named" + name); 
    auto it = list.find(name); 
    assert(it->second.type == Property::EPTexture, "Wrong Value's Type"); 
    return it->second.value.texture_value; 
}

std::shared_ptr<Texture>
PropertyList::getTexture(
    const std::string& name, 
    const float& defaultValue) const
{
    if (list.find(name) == list.end())
    {
        return std::make_shared<Texture>(defaultValue);
    }
    auto it = list.find(name);
    assert(it->second.type == Property::EPTexture || 
           it->second.type == Property::EPFloat, "Wrong Value's Type");
    if (it->second.type == Property::EPFloat)
    {
        return std::shared_ptr<Texture>(new Texture(it->second.value.float_value));
    }
    else
    {
        return it->second.value.texture_value;
    }    
}

std::shared_ptr<Texture>
PropertyList::getTexture(
    const std::string& name,
    const Spectrum& defaultValue) const
{
    if (list.find(name) == list.end())
    {
        return std::make_shared<Texture>(defaultValue);
    }
    auto it = list.find(name);
    assert(it->second.type == Property::EPTexture || 
           it->second.type == Property::EPColor, "Wrong Value's Type");
    if (it->second.type == Property::EPColor)
    {
        return std::shared_ptr<Texture>(new Texture(it->second.value.color_value));
    }
    else
    {
        return it->second.value.texture_value;
    }
}

bool PropertyList::findTexture(
    const std::string& name) const
{
    if (list.find(name) != list.end()) {        
        auto it = list.find(name); 
        assert(it->second.type == Property::EPTexture || 
               it->second.type == Property::EPFloat || 
               it->second.type == Property::EPColor, "Wrong Value's Type");
        return true; 
    }
    else {
        return false;
    }
}
