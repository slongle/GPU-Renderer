#pragma once

#include "renderer/fwd.h"
#include "renderer/material.h"
#include "renderer/sampling.h"
#include "renderer/aabb.h"

class Triangle {
public:
    Triangle() {}

    Triangle(
        const float3& p0, const float3& p1, const float3& p2)
        :m_p0(p0), m_p1(p1), m_p2(p2),
        m_has_n(false)
    {}

    Triangle(
        const float3& p0, const float3& p1, const float3& p2,
        const float3& n0, const float3& n1, const float3& n2)
        :m_p0(p0), m_p1(p1), m_p2(p2),
        m_n0(n0), m_n1(n1), m_n2(n2),
        m_has_n(true)
    {}

    Triangle(
        const float3& p0, const float3& p1, const float3& p2,        
        const Material& material)
        :m_p0(p0), m_p1(p1), m_p2(p2),        
        m_has_n(false),
        m_material(material)
    {}

    Triangle(
        const float3& p0, const float3& p1, const float3& p2, 
        const float3& n0, const float3& n1, const float3& n2,
        const Material& material) 
        :m_p0(p0), m_p1(p1), m_p2(p2), 
         m_n0(n0), m_n1(n1), m_n2(n2),
         m_has_n(true),
         m_material(material)
    {}

    HOST_DEVICE
    AABB worldBound() const
    {
        return Union(AABB(m_p0, m_p1), m_p2);
    }

    HOST_DEVICE
    void sample(
        LightSample* record, 
        const float2 s) const
    {
        float2 uv = UniformSampleTriangle(s);
        record->m_p = m_p0* (1 - uv.x - uv.y) + m_p1 * uv.x + m_p2 * uv.y;
        float3 n = cross(m_p1 - m_p0, m_p2 - m_p0);
        record->m_normal_g = normalize(n);
        if (m_has_n)
        {
            record->m_normal_s = m_n0 * (1 - uv.x - uv.y) + m_n1 * uv.x + m_n2 * uv.y;            
        }
        else
        {
            record->m_normal_s = record->m_normal_g;
        }
        record->m_normal_s = record->m_normal_g;
        record->m_pdf = 2.f / length(n);        
    }    

    float3 m_p0, m_p1, m_p2;
    float3 m_n0, m_n1, m_n2;
    Material m_material;
    bool m_has_n;
};