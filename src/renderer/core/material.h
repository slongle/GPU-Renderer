#pragma once
#ifndef __MATERIAL_H
#define __MATERIAL_H

#include "renderer/core/fwd.h"
#include "renderer/core/parameterset.h"
#include "renderer/core/spectrum.h"

class Material {
public:
    Material(Spectrum Kd);

    Spectrum m_Kd;
};

inline
std::shared_ptr<Material>
CreateMatteMaterial(
    const ParameterSet& param);

inline
Material::Material(Spectrum Kd) :m_Kd(Kd)
{
}

inline
std::shared_ptr<Material>
CreateMatteMaterial(
    const ParameterSet& param)
{
    std::vector<Float> rgbs = param.GetSpectrum("Kd");
    Spectrum Kd(rgbs[0], rgbs[1], rgbs[2]);
    return std::make_shared<Material>(Kd);
}



#endif // !__MATERIAL_H
