#include "specular.h"

SpecularReflect::SpecularReflect(
    const Spectrum& r,
    const bool& conductor, 
    const Spectrum& etaI, 
    const Spectrum& etaT, 
    const Spectrum& k)
    : m_r(r), m_conductor(conductor), m_conductorEtaI(etaI), m_conductorEtaT(etaT), m_conductorK(k)
{
}

SpecularReflect::SpecularReflect(
    const Spectrum& r, 
    const bool& conductor, 
    const float& etaI, 
    const float& etaT)
    : m_r(r), m_conductor(conductor), m_dielectricEtaI(etaI), m_dielectricEtaT(etaT)
{
}

Spectrum SpecularReflect::f(
    const float2 s, 
    const float3 wo, 
    const float3 wi) const
{
    return Spectrum(0.f);
}

void SpecularReflect::pdf(
    const float2 s, 
    const float3 wo, 
    const float3 wi, 
    float* pdf) const
{
    *pdf = 0.f;
}

void SpecularReflect::sample(
    const float2 s, 
    const float3 wo, 
    float3* wi, 
    Spectrum* f, 
    float* pdf) const
{
    *wi = make_float3(-wo.x, -wo.y, wo.z);
    *pdf = 1.f;
    Spectrum F;
    if (m_conductor) {
        F = FrConductor(fabsf(wi->z), m_conductorEtaI, m_conductorEtaT, m_conductorK);
    }
    else {
        F = Spectrum(FrDielectric(wi->z, m_dielectricEtaI, m_dielectricEtaT));
    }
    *f = F * m_r / fabsf(wi->z);
}

