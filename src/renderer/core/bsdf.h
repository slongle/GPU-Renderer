#pragma once
#ifndef __BSDF_H
#define __BSDF_H

#include "renderer/core/fwd.h"
#include "renderer/core/geometry.h"
#include "renderer/core/spectrum.h"
#include "renderer/core/interaction.h"

class BSDF {
public:
    __device__ __host__ BSDF(const Interaction& inter) {}

    __device__ __host__ Vector3f WorldToLocal(const Vector3f& v) const;
    __device__ __host__ Vector3f LocalToWorld(const Vector3f& v) const;
    __device__ __host__ Float AbsCosTheta(const Vector3f& v) const;

    __device__ __host__ Spectrum Sample(const Vector3f& worldWo, Vector3f* worldWi, Float* pdf) const;

    // Lambertian Reflection
    Spectrum m_r;
};

inline __device__ __host__ 
Vector3f BSDF::WorldToLocal(const Vector3f& v) const
{
    return Vector3f();
}

inline __device__ __host__ 
Vector3f BSDF::LocalToWorld(const Vector3f& v) const
{
    return Vector3f();
}

inline __device__ __host__
Float BSDF::AbsCosTheta(const Vector3f& v) const {
    return v.z;
}

inline __device__ __host__ 
Spectrum BSDF::Sample(const Vector3f& worldWo, Vector3f* worldWi, Float* pdf) const
{
    return Spectrum(0);
}

#endif // __BSDF_H