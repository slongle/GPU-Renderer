#pragma once

#include "fwd.h"

struct LightSample
{
    HOST_DEVICE
    LightSample() {}

    uint32 m_light_id;
    float3 m_p;
    float3 m_normal_s;
    float3 m_normal_g;
    float  m_pdf;
};

struct BSDFSample 
{
    HOST_DEVICE
    BSDFSample() {}

    float3   m_wo;
    float3   m_wi;
    Spectrum m_f;
    float    m_pdf;
};

inline HOST_DEVICE
uint32 InitRandom(
    uint32 val0,
    uint32 val1,
    uint32 backoff = 16)
{
    uint32 v0 = val0, v1 = val1, s0 = 0;

    for (uint32 n = 0; n < backoff; n++)
    {
        s0 += 0x9e3779b9;
        v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
        v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
    }
    return v0;
}

inline HOST_DEVICE
float NextRandom(
    uint32& seed)
{
    seed = (1664525u * seed + 1013904223u);
    return float(seed & 0x00FFFFFF) / float(0x01000000);
}

inline HOST_DEVICE
float2 UniformSampleTriangle(const float2 s) 
{    
    float a = sqrt(s.x);
    return make_float2(1 - a, s.y * a);
}

inline HOST_DEVICE
float3 CosineSampleHemisphere(const float2 u)
{   
    float a = sqrt(u.x);
    float b = PI * 2 * u.y;
    float z = sqrt(max(0.f, 1 - u.x));
    return make_float3(a * cos(b), a * sin(b), z);
}