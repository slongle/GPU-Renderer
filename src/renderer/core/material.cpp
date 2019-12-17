#include "material.h"

Material::Material(
    MaterialType type, 
    Spectrum Kd) 
    : m_type(type), m_Kd(Kd)
{
}

std::shared_ptr<Material>
CreateMatteMaterial(
    const ParameterSet& param)
{
    std::vector<Float> rgbs = param.GetSpectrum("Kd");
    Spectrum Kd(rgbs);
    return std::make_shared<Material>(Material::DIFFUSE_MATERIAL, Kd);
}


