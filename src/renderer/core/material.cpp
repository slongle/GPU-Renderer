#include "material.h"

Material::Material(Spectrum Kd) :m_Kd(Kd)
{
}

std::shared_ptr<Material> 
CreateMatteMaterial(const ParameterSet& param)
{
    std::vector<Float> rgbs = param.GetSpectrum("Kd");
    Spectrum Kd(rgbs[0], rgbs[1], rgbs[2]);
    return std::make_shared<Material>(Kd);
}

