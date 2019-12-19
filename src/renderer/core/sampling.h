#pragma once
#ifndef __SAMPLING_H
#define __SAMPLING_H

#include "renderer/core/fwd.h"

inline __device__ __host__
unsigned int InitRandom(
    unsigned int val0, 
    unsigned int val1, 
    unsigned int backoff = 16)
{
    unsigned int v0 = val0, v1 = val1, s0 = 0;

    for (unsigned int n = 0; n < backoff; n++)
    {
        s0 += 0x9e3779b9;
        v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
        v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
    }
    return v0;
}

inline __device__ __host__
Float NextRandom(
    unsigned int& seed)
{
    seed = (1664525u * seed + 1013904223u);
    return float(seed & 0x00FFFFFF) / float(0x01000000);
}

inline __device__ __host__
void CoordinateSystem(const Normal3f& n, Vector3f* s, Vector3f* t) {
    if (abs(n.x) > abs(n.y)) {
        Float invLen = Float(1) / sqrt(n.x * n.x + n.z * n.z);
        *s = Vector3f(n.z * invLen, 0, -n.x * invLen);
    }
    else {
        Float invLen = Float(1) / sqrt(n.y * n.y + n.z * n.z);
        *s = Vector3f(0, n.z * invLen, -n.y * invLen);
    }
    *t = Cross(*s, n);
}

inline __device__ __host__
Vector3f LocalToWorld(const Vector3f& v, const Normal3f& n, const Vector3f& s, const Vector3f t) {
    return Normalize(s * v.x + t * v.y + n * v.z);
}

inline __device__ __host__
Vector3f WorldToLocal(const Vector3f& v, const Normal3f& n, const Vector3f& s, const Vector3f t) {
    return Normalize(Vector3f(Dot(v, s), Dot(v, t), Dot(v, n)));
}

inline __device__ __host__
Vector3f UniformSampleHemisphere(unsigned int& seed)
{
    Float u = NextRandom(seed), v = NextRandom(seed);    
    Float a = sqrt(max((Float)0, 1 - u * u));
    Float b = Pi * 2 * v;
    return Vector3f(a * cos(b), a * sin(b), u);
}

inline __device__ __host__
Float UniformSampleHemispherePdf()
{
    return Inv2Pi;
}

inline __device__ __host__
Vector3f CosineSampleHemisphere(unsigned int& seed)
{
    Float u = NextRandom(seed), v = NextRandom(seed);
    Float a = sqrt(u);
    Float b = Pi * 2 * v;
    Float z = sqrt(max(Float(0), 1 - u));
    return Vector3f(a * cos(b), a * sin(b), z);
}

inline __device__ __host__
Float CosineSampleHemispherePdf(Float cosTheta)
{
    return InvPi * cosTheta;
}

inline __device__ __host__
Point2f UniformSampleTriangle(unsigned int& seed) {
    Float u = NextRandom(seed), v = NextRandom(seed);
    Float a = sqrt(u);
    return Point2f(1 - a, v * a);
}

#endif // __SAMPLING_H