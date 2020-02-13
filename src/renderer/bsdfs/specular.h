#pragma once

#include "renderer/spectrum.h"
#include "renderer/bsdfs/optical.h"

class SpecularReflect{
public:
    HOST_DEVICE
    SpecularReflect() {}

    // Conductor
    HOST_DEVICE
    SpecularReflect(
        const Spectrum& r,
        const bool& conductor,
        const Spectrum& etaI,
        const Spectrum& etaT,
        const Spectrum& k);

    // Dielectric
    HOST_DEVICE
    SpecularReflect(
        const Spectrum& r,
        const bool& conductor,
        const float& etaI,
        const float& etaT);

    /// evaluate f = BSDF * cos, BSDF = color / PI
    ///
    HOST_DEVICE
    Spectrum f(
        const float2    s,
        const float3    wo,
        const float3    wi) const;

    /// evaluate pdf
    ///
    HOST_DEVICE
    void pdf(
        const float2    s,
        const float3    wo,
        const float3    wi,
        float* pdf) const;

    /// sample wi, evaluate f and pdf
    ///
    HOST_DEVICE
    void sample(
        const float2    s,
        const float3    wo,
        float3* wi,
        Spectrum* f,
        float* pdf) const;

    // Global
    Spectrum m_r;
    bool m_conductor;
    // Conduct Fresnel
    Spectrum m_conductorEtaI, m_conductorEtaT, m_conductorK;
    // Dielectric Fresnel
    float m_dielectricEtaI, m_dielectricEtaT;
};

class FresnelSpecular 
{
public:
    HOST_DEVICE
    FresnelSpecular() {}

    HOST_DEVICE
    FresnelSpecular(
        const Spectrum& t,
        const Spectrum& r,
        const float& etaA,
        const float& etaB);

    /// evaluate eval = BSDF * cos, BSDF = color / PI
    ///
    HOST_DEVICE
    Spectrum eval(
        const float3    wo,
        const float3    wi) const;

    /// evaluate pdf
    ///
    HOST_DEVICE
    void pdf(
        const float3    wo,
        const float3    wi,
        float* pdf) const;

    /// sample wi, evaluate f and pdf
    ///
    HOST_DEVICE
    void sample(
        const float2    s,
        const float3    wo,
        float3* wi,
        Spectrum* f,
        float* pdf) const;

    // Global
    Spectrum m_t, m_r;
    // Dielectric Fresnel
    float m_etaA, m_etaB;
};

inline HOST_DEVICE
FresnelSpecular::FresnelSpecular(
    const Spectrum& t,
    const Spectrum& r,
    const float& etaA,
    const float& etaB)
    : m_t(t), m_r(r), m_etaA(etaA), m_etaB(etaB)
{
}

inline HOST_DEVICE
Spectrum FresnelSpecular::eval(
    const float3 wo,
    const float3 wi) const
{
    return Spectrum(0.f);
}

inline HOST_DEVICE
void FresnelSpecular::pdf(
    const float3 wo,
    const float3 wi,
    float* pdf) const
{
    *pdf = 0.f;
}

inline HOST_DEVICE
void FresnelSpecular::sample(
    const float2 s,
    const float3 wo,
    float3* wi,
    Spectrum* f,
    float* pdf) const
{
    float F = FrDielectric(wo.z, m_etaA, m_etaB);    
    float u = s.x;
    if (u < F) {
        // Compute specular reflection for _FresnelSpecular_

        // Compute perfect specular reflection direction
        *wi = make_float3(-wo.x, -wo.y, wo.z);

        *pdf = F;
        *f = m_r * F;
    }
    else {
        // Compute specular transmission for _FresnelSpecular_

        // Figure out which $\eta$ is incident and which is transmitted
        bool entering = wo.z > 0;
        float etaI = entering ? m_etaA : m_etaB;
        float etaT = entering ? m_etaB : m_etaA;

        // Compute ray direction for specular transmission        
        float3 n = make_float3(0, 0, 1);
        if (wo.z < 0) n = -n;
        if (!Refract(wo, n, etaI / etaT, wi))
        {
            *f = Spectrum(0.f);
            return;
        }
        Spectrum ft = m_t * (1 - F);

        ft *= (etaI * etaI) / (etaT * etaT);

        *pdf = 1 - F;
        *f = ft;
    }
}
