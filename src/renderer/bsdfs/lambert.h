#pragma once

#include "renderer/bsdfs/optical.h"

inline HOST_DEVICE
void LambertReflectEval(
    const float3& wo,
    const float3& wi,
    Spectrum* f,
    const Spectrum& color)
{
    *f = 0.f;

    if (!SameHemisphere(wo, wi)) return;    
    *f = color * INV_PI * fabsf(wi.z);
    
}

inline HOST_DEVICE
void LambertReflectPdf(
    const float3& wo,
    const float3& wi,
    float* pdf,
    const Spectrum& color)
{
    *pdf = 0.f;

    if (!SameHemisphere(wo, wi)) return;
    *pdf = fabsf(wi.z) * INV_PI;
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
    *f = *pdf = 0.f;

    *wi = CosineSampleHemisphere(u);
    if (wo.z < 0) wi->z *= -1;
    *f = color * INV_PI * fabsf(wi->z);
    *pdf = fabsf(wi->z) * INV_PI;
}