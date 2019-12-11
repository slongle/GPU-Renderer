#pragma once
#ifndef __AREALIGHT_H
#define __AREALIGHT_H

#include "renderer/core/light.h"

class AreaLight : public Light {
public:

};

std::shared_ptr<AreaLight>
CreateAreaLight(
    const ParameterSet& param,
    const std::shared_ptr<Shape>& s);

#endif // !__AREALIGHT_H
