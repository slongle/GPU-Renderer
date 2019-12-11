#pragma once
#ifndef __PRIMITIVE_H
#define __PRIMITIVE_H

#include "renderer/core/fwd.h"
#include "renderer/core/shape.h"
#include "renderer/core/material.h"
#include "renderer/light/arealight.h"

class Primitive {
public:
    Primitive(
        const std::shared_ptr<Shape>& shape,
        const std::shared_ptr<Material>& material,
        const std::shared_ptr<AreaLight>& areaLight) 
        : m_shape(shape), m_material(material), m_areaLight(areaLight) {}

    std::shared_ptr<Shape> m_shape;
    std::shared_ptr<Material> m_material;
    std::shared_ptr<AreaLight> m_areaLight;
};

#endif // !__PRIMITIVE_H
