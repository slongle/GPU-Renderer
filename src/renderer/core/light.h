#pragma once
#ifndef __LIGHT_H
#define __LIGHT_H

#include "renderer/core/fwd.h"
#include "renderer/core/parameterset.h"
#include "renderer/core/spectrum.h"

class Light {
public:
    enum LightType{
        AREA_LIGHT = 0,
    };

    __device__ __host__ Light() {}
    __device__ __host__ Light(LightType type, const Spectrum& L, int shapeID);


// Global
    LightType m_type;
// Area Light
    int m_shapeID;
    Spectrum m_L;
};

std::shared_ptr<Light>
CreateAreaLight(
    const ParameterSet& params,
    int shapeID);

#endif // !__LIGHT_H
