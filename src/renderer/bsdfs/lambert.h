#pragma once

#include "renderer/bsdfs/optical.h"

inline HOST_DEVICE
void LambertReflectEval(
    const float3& wo,
    const float3& wi,
    Spectrum* f,
    const Spectrum& color)
{
    if (SameHemisphere(wo, wi))
    {
        *f = color * INV_PI * fabsf(wi.z);
    }
    else
    {
        *f = 0.f;
    }
}

inline HOST_DEVICE
void LambertReflectPdf(
    const float3& wo,
    const float3& wi,
    float* pdf,
    const Spectrum& color)
{
    if (SameHemisphere(wo, wi))
    {
        *pdf = fabsf(wi.z) * INV_PI;
    }
    else
    {
        *pdf = 0.f;
    }
}

inline HOST_DEVICE
void LambertReflectSample(
    const float2& u,
    const float3& wo,
    float3* wi,
    Spectrum* f,
    float* pdf,
    const Spectrum& color)
{
    *wi = CosineSampleHemisphere(u);
    *f = color * INV_PI * fabsf(wi->z);
    *pdf = fabsf(wi->z) * INV_PI;
}