#pragma once
#ifndef __INTERACTION_H
#define __INTERACTION_H

#include "renderer/core/geometry.h"

class Interaction {
public:
    __host__ __device__
    Interaction() {}

    __host__ __device__
    Ray SpawnRayTo(const Interaction& inter) const {
        Vector3f d = inter.m_p - m_p;
        return Ray(m_p, d, Epsilon, 1.f - ShadowEpsilon);
        /*
        Point3f origin = m_p + (inter.m_p - m_p) * Epsilon;
        Point3f target = inter.m_p + (origin - inter.m_p) * Epsilon;
        Vector3f d = target - origin;
        return Ray(origin, Normalize(d), d.Length() - Epsilon);
        */
    }

    __host__ __device__
    Ray SpawnRay(const Vector3f& d) const {
        Point3f origin = m_p + d * Epsilon;
        return Ray(origin, d);
    }

// Global
    int m_primitiveID;
// Surface
    Vector3f m_wo, m_wi;
    Point3f m_p;
    Normal3f m_shadingN, m_geometryN;
    Point2f m_uv;
    BSDF* m_bsdf;
// Medium
};

#endif // !__INTERACTION_H
