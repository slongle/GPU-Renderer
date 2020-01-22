#pragma once

#include "renderer/fwd.h"

struct AABB
{
    HOST_DEVICE
    AABB() { m_min = make_float3(INFINITY), m_max = make_float3(-INFINITY); }

    HOST_DEVICE
    AABB(const float3& p) : m_min(p), m_max(p) {}

    HOST_DEVICE
    AABB(
        const float3& p,
        const float3& q)
        : m_min(fminf(p, q)), m_max(fmaxf(p, q)) {}
    
    HOST_DEVICE
    float3 operator[] (const int idx) const { return idx == 0 ? m_min : m_max; }

    HOST_DEVICE
    float3 centroid() const { return (m_min + m_max) * 0.5f; }

    float3 m_min, m_max;
};

inline HOST_DEVICE
AABB Union(
    const AABB&   box,
    const float3& p)
{
    AABB ret;
    ret.m_min = fminf(box.m_min, p);
    ret.m_max = fmaxf(box.m_max, p);
    return ret;
}

inline HOST_DEVICE
AABB Union(
    const AABB& box1,
    const AABB& box2)
{
    AABB ret;
    ret.m_min = fminf(box1.m_min, box2.m_min);
    ret.m_max = fmaxf(box1.m_max, box2.m_max);
    return ret;
}