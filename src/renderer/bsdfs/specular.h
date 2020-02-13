#pragma once

#include "renderer/bsdfs/optical.h"

inline HOST_DEVICE
void FresnelSpecularEval(
    const float3& wo,
    const float3& wi,
    Spectrum* f,
    const Spectrum& color,
    const float& ior)
{
    *f = Spectrum(0);
}

inline HOST_DEVICE
void FresnelSpecularPdf(
    const float3& wo,
    const float3& wi,
    float* pdf,
    const Spectrum& color, 
    const float& ior)
{
    *pdf = 0;
}

inline HOST_DEVICE
void FresnelSpecularSample(
    const float2& u,
    const float3& wo,
    float3* wi,
    Spectrum* f,
    float* pdf,
    const Spectrum& color, 
    const float& ior)
{
    float F = FrDielectric(wo.z, 1, ior);
    if (u.x < F) {
        // Compute specular reflection for _FresnelSpecular_

        // Compute perfect specular reflection direction
        *wi = make_float3(-wo.x, -wo.y, wo.z);

        *pdf = F;
        *f = color * F;
    }
    else {
        // Compute specular transmission for _FresnelSpecular_

        // Figure out which $\eta$ is incident and which is transmitted
        bool entering = wo.z > 0;
        float etaI = entering ? 1 : ior;
        float etaT = entering ? ior : 1;

        // Compute ray direction for specular transmission        
        float3 n = make_float3(0, 0, 1);
        if (wo.z < 0) n = -n;
        if (!Refract(wo, n, etaI / etaT, wi))
        {
            *f = Spectrum(0.f);
            return;
        }
        Spectrum ft = color * (1 - F);

        ft *= (etaI * etaI) / (etaT * etaT);

        *pdf = 1 - F;
        *f = ft;
    }
}