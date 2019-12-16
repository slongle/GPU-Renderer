#include "light.h"

Light::Light(
    LightType type, 
    const Spectrum& L, 
    int shapeID)
    : m_type(type), m_L(L), m_shapeID(shapeID)
{
}

std::shared_ptr<Light>
CreateAreaLight(
    const ParameterSet& params,
    int shapeID)
{
    std::vector<Float> rgbs = params.GetSpectrum("L");
    Spectrum L(rgbs);
    return std::make_shared<Light>(Light::AREA_LIGHT, L, shapeID);
}

