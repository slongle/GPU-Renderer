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
    bool     m_specular;
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

inline HOST_DEVICE
bool SameHemisphere(const float3& w, const float3& wp) {
    return w.z * wp.z > 0;
}

inline __device__ __host__
float Clamp(float a, float l, float r) {
    if (a > r) return r;
    else if (a < l) return l;
    else return a;
}

inline __device__ __host__
float CosTheta(const float3& v) { return v.z; }
inline __device__ __host__
float AbsCosTheta(const float3& v) { return abs(v.z); }
inline __device__ __host__
float Cos2Theta(const float3& v) { return v.z * v.z; }

inline __device__ __host__
float Sin2Theta(const float3& w) {
    return max((float)0, (float)1 - Cos2Theta(w));
}
inline __device__ __host__
float SinTheta(const float3& w) { return sqrt(Sin2Theta(w)); }

inline __device__ __host__
float TanTheta(const float3& w) { return SinTheta(w) / CosTheta(w); }
inline __device__ __host__
float Tan2Theta(const float3& w) {
    return Sin2Theta(w) / Cos2Theta(w);
}

inline __device__ __host__
float CosPhi(const float3& w) {
    float sinTheta = SinTheta(w);
    if (sinTheta == 0) return 1;
    else return Clamp(w.x / sinTheta, -1, 1);
}
inline __device__ __host__
float Cos2Phi(const float3& w) { return CosPhi(w) * CosPhi(w); }

inline __device__ __host__
float SinPhi(const float3& w) {
    float sinTheta = SinTheta(w);
    if (sinTheta == 0) return 0;
    else return Clamp(w.y / sinTheta, -1, 1);
}
inline __device__ __host__
float Sin2Phi(const float3& w) { return SinPhi(w) * SinPhi(w); }


class Frame
{
public:
    HOST_DEVICE
    Frame() {}
    HOST_DEVICE
    Frame(
        const float3& normal_g,
        const float3& normal_s)
        : m_normal_g(normal_g), m_normal_s(normal_s)
    {
        const float3 n = normal_s;
        if (fabsf(n.x) > fabsf(n.y)) {
            float inv_len = 1.f / sqrtf(n.x * n.x + n.z * n.z);
            m_s = make_float3(n.z * inv_len, 0, -n.x * inv_len);
        }
        else {
            float inv_len = 1.f / sqrtf(n.y * n.y + n.z * n.z);
            m_s = make_float3(0, n.z * inv_len, -n.y * inv_len);
        }
        m_t = cross(m_s, n);
    }

    HOST_DEVICE
    float3 localToWorld(
        const float3& v) const
    {
        return normalize(m_s * v.x + m_t * v.y + m_normal_s * v.z);
    }

    HOST_DEVICE
    float3 worldToLocal(
        const float3& v) const
    {
        return normalize(make_float3(dot(v, m_s), dot(v, m_t), dot(v, m_normal_s)));
    }

    float3 m_normal_g;
    float3 m_normal_s;
    float3 m_s;
    float3 m_t;
};