#pragma once
#ifndef __PRIMITIVE_H
#define __PRIMITIVE_H

#include "renderer/core/fwd.h"
#include "renderer/core/triangle.h"
#include "renderer/core/material.h"
#include "renderer/core/light.h"

class Primitive {
public:
    Primitive() {}

    Primitive(
        int shapeID,
        int materialID,
        int lightID) 
        : m_shapeID(shapeID), m_materialID(materialID), m_lightID(lightID) {}

    int m_shapeID;
    int m_materialID;
    int m_lightID;
};

#endif // !__PRIMITIVE_H
