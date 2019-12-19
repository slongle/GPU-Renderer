#pragma once
#ifndef __MATERIAL_H
#define __MATERIAL_H

#include "renderer/core/fwd.h"
#include "renderer/core/parameterset.h"
#include "renderer/core/spectrum.h"
#include "renderer/core/interaction.h"
#include "renderer/core/sampling.h"
#include "renderer/core/bsdf.h"

class Material {
public:
    enum MaterialType {
        DIFFUSE_MATERIAL = 0
    };

    Material(MaterialType type, Spectrum Kd);
   
    // Global
    MaterialType m_type;

    // Diffuse
    Spectrum m_Kd;
};

std::shared_ptr<Material>
CreateMatteMaterial(
    const ParameterSet& param);



#endif // !__MATERIAL_H
