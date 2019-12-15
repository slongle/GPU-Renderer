#pragma once
#ifndef __LIGHT_H
#define __LIGHT_H

#include "renderer/core/fwd.h"
#include "renderer/core/parameterset.h"

class Light {
public:
    

    enum {
        AREA_LIGHT = 0,
    };

    int m_shapeID;
};


inline
std::shared_ptr<Light>
CreateAreaLight(
    const ParameterSet& params,
    int shapeID);


inline
std::shared_ptr<Light>
CreateAreaLight(
    const ParameterSet& params,
    int shapeID)
{
    return std::shared_ptr<Light>();
}

#endif // !__LIGHT_H
