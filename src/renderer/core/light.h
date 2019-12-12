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

std::shared_ptr<Light>
CreateAreaLight(
    const ParameterSet& params,
    int shapeID);

#endif // !__LIGHT_H
