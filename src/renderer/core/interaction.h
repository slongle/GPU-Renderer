#pragma once
#ifndef __INTERACTION_H
#define __INTERACTION_H

#include "renderer/core/fwd.h"

class Interaction {
public:
    __host__ __device__
    Interaction() {}

// Global
    int m_primitiveID;
// Surface
    Vector3f m_wo, m_wi;
    Point3f m_p;
    Normal3f m_shadingN, m_geometryN;
    Point2f m_uv;
// Medium
};

#endif // !__INTERACTION_H
