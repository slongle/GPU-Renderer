#pragma once

#include "renderer/bsdfs/optical.h"
#include "renderer/bsdfs/microfacet.h"

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

inline HOST_DEVICE
void FresnelBlendEval(
    const float3& wo,
    const float3& wi,
    Spectrum* f,
    const Spectrum& Rd,
    const Spectrum& Rs,
    const float& alpha_x,
    const float& alpha_y)
{    
    using namespace TrowbridgeReitzDistribution;
    *f = Spectrum(0.f);

    Spectrum diffuse = (28.f / (23.f * PI)) * Rd * (Spectrum(1.f) - Rs) *
        (1 - pow5(1 - .5f * AbsCosTheta(wi))) *
        (1 - pow5(1 - .5f * AbsCosTheta(wo)));
    float3 wh = wi + wo;
    if (wh.x == 0 && wh.y == 0 && wh.z == 0) return;
    wh = normalize(wh);
    Spectrum specular = D(wh, alpha_x, alpha_y) * AbsCosTheta(wi) /
        (4 * fabsf(dot(wi, wh)) * max(AbsCosTheta(wi), AbsCosTheta(wo))) *
        SchlickFresnel(Rs, dot(wi, wh));
    *f = diffuse + specular;
}

inline HOST_DEVICE
void FresnelBlendPdf(
    const float3& wo,
    const float3& wi,
    float* pdf,
    const Spectrum& Rd,
    const Spectrum& Rs,
    const float& alpha_x,
    const float& alpha_y)
{
    using namespace TrowbridgeReitzDistribution;
    *pdf = 0;

    if (!SameHemisphere(wo, wi)) return;
    float3 wh = normalize(wo + wi);
    float pdf_wh = Pdf(wo, wh, alpha_x, alpha_y);
    *pdf = .5f * (AbsCosTheta(wi) * INV_PI + pdf_wh / (4 * dot(wo, wh)));
}

inline HOST_DEVICE
void FresnelBlendSample(
    const float2& u,
    const float3& wo,
    float3* wi,
    Spectrum* f,
    float* pdf,
    const Spectrum& Rd,
    const Spectrum& Rs,
    const float& alpha_x,
    const float& alpha_y)
{
    using namespace TrowbridgeReitzDistribution;
    *f = Spectrum(0.f);
    *pdf = 0.f;

    float2 v = u;
    if (v.x < .5) {
        v.x = min(2 * v.x, 1 - 1e-4);
        // Cosine-sample the hemisphere, flipping the direction if necessary
        *wi = CosineSampleHemisphere(u);
        if (wo.z < 0) wi->z *= -1;
    }
    else {
        v.x = min(2 * (v.x - .5f), 1 - 1e-4);
        // Sample microfacet orientation $\wh$ and reflected direction $\wi$
        float3 wh = Sample_wh(wo, v, alpha_x, alpha_y);
        *wi = Reflect(wo, wh);
        if (!SameHemisphere(wo, *wi)) return;
    }
    FresnelBlendPdf(wo, *wi, pdf, Rd, Rs, alpha_x, alpha_y);
    return FresnelBlendEval(wo, *wi, f, Rd, Rs, alpha_x, alpha_y);
}