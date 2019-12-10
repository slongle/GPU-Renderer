#pragma once
#ifndef __PRIMITIVE_H
#define __PRIMITIVE_H

#include "renderer/core/fwd.h"
#include "renderer/core/shape.h"
#include "renderer/core/material.h"

class Primitive {
public:
    std::shared_ptr<Shape> m_shape;
    std::shared_ptr<Material> m_material;
};

#endif // !__PRIMITIVE_H
