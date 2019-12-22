#pragma once
#ifndef __BSDF_H
#define __BSDF_H

#include "renderer/core/fwd.h"
#include "renderer/core/geometry.h"
#include "renderer/core/spectrum.h"
#include "renderer/core/interaction.h"
#include "renderer/core/sampling.h"
#include "renderer/core/microfacet.h"
#include "renderer/core/optic.h"

class LambertReflectBSDF {
public:
    LambertReflectBSDF() {}
    LambertReflectBSDF(const Spectrum& r);

    Spectrum Sample(const Vector3f& wo, Vector3f* wi, Float* pdf, unsigned int& seed) const;
    Spectrum F(const Vector3f& wo, const Vector3f& wi) const;
    Spectrum F(const Vector3f& wo, const Vector3f& wi, Float* pdf) const;

    Spectrum m_r;
};

class GGXSmithReflectBSDF {
public:
    GGXSmithReflectBSDF() {}
    GGXSmithReflectBSDF(
        const Spectrum& r, 
        const bool& conductor,
        const Spectrum& etaI, 
        const Spectrum& etaT,
        const Spectrum& k,
        const Float& uroughness,
        const Float& vroughness);

    GGXSmithReflectBSDF(
        const Spectrum& r,
        const bool& conductor,
        const Float& etaI,
        const Float& etaT,
        const Float& uroughness,
        const Float& vroughness);

    Spectrum Sample(const Vector3f& wo, Vector3f* wi, Float* pdf, unsigned int& seed) const;
    Spectrum F(const Vector3f& wo, const Vector3f& wi) const;
    Spectrum F(const Vector3f& wo, const Vector3f& wi, Float* pdf) const;

    // Global
    Spectrum m_r;
    bool m_conductor;
    // Conduct Fresnel
    Spectrum m_conductorEtaI, m_conductorEtaT, m_conductorK;
    // Dielectric Fresnel
    Float m_dielectricEtaI, m_dielectricEtaT;
    // GGX
    GGXDistribution m_distribution;
};


inline __host__ __device__
LambertReflectBSDF::LambertReflectBSDF(
    const Spectrum& r)
    : m_r(r)
{
}

inline __host__ __device__
Spectrum LambertReflectBSDF::Sample(
    const Vector3f& wo, 
    Vector3f* wi, 
    Float* pdf, 
    unsigned int& seed) const
{
    *wi = CosineSampleHemisphere(seed);
    *pdf = CosineSampleHemispherePdf(AbsCosTheta(*wi));
    return m_r * InvPi * AbsCosTheta(*wi);
}

inline __host__ __device__
Spectrum LambertReflectBSDF::F(
    const Vector3f& wo, 
    const Vector3f& wi) const
{
    return m_r * InvPi * AbsCosTheta(wi);
}

inline __host__ __device__
Spectrum LambertReflectBSDF::F(
    const Vector3f& wo, 
    const Vector3f& wi, 
    Float* pdf) const
{
    *pdf = CosineSampleHemispherePdf(AbsCosTheta(wi));
    return m_r * InvPi * AbsCosTheta(wi);
}

inline
GGXSmithReflectBSDF::GGXSmithReflectBSDF(
    const Spectrum& r,
    const bool& conductor,
    const Spectrum& etaI,
    const Spectrum& etaT,
    const Spectrum& k,
    const Float& uroughness,
    const Float& vroughness)
    : m_r(r), m_conductor(conductor), m_conductorEtaI(etaI), m_conductorEtaT(etaT), m_conductorK(k), 
      m_distribution(uroughness, vroughness)
{
}

inline
GGXSmithReflectBSDF::GGXSmithReflectBSDF(
    const Spectrum& r,
    const bool& conductor,
    const Float& etaI,
    const Float& etaT,
    const Float& uroughness,
    const Float& vroughness)
    : m_r(r), m_conductor(conductor), m_dielectricEtaI(etaI), m_dielectricEtaT(etaT), 
      m_distribution(uroughness, vroughness)
{
}

inline __host__ __device__
Spectrum GGXSmithReflectBSDF::Sample(
    const Vector3f& wo, 
    Vector3f* wi, 
    Float* pdf, 
    unsigned int& seed) const
{
    // Sample microfacet orientation $\wh$ and reflected direction $\wi$    
    if (wo.z == 0) return Spectrum(0.f);    
    Vector3f wh = m_distribution.Sample_wh(wo, seed);
    *wi = Reflect(wo, wh);
    if (!SameHemisphere(wo, *wi)) return Spectrum(0.f);

    // Compute PDF of _wi_ for microfacet reflection
    *pdf = m_distribution.Pdf(wo, wh) / (4 * Dot(wo, wh));
    return F(wo, *wi);
}

inline __host__ __device__
Spectrum GGXSmithReflectBSDF::F(
    const Vector3f& wo, 
    const Vector3f& wi) const
{
    //return 0;
    Float cosThetaO = AbsCosTheta(wo), cosThetaI = AbsCosTheta(wi);
    Vector3f wh = wi + wo;
    // Handle degenerate cases for microfacet reflection
    if (cosThetaI == 0 || cosThetaO == 0) return Spectrum(0.f);
    if (wh.x == 0 && wh.y == 0 && wh.z == 0) return Spectrum(0.f);
    wh = Normalize(wh);    
    Spectrum F;
    if (m_conductor) {
        F = FrConductor(abs(Dot(wi, wh)), m_conductorEtaI, m_conductorEtaT, m_conductorK);
    }
    else {
        F = FrDielectric(Dot(wi, wh), m_dielectricEtaI, m_dielectricEtaT);
    }
    return m_r * m_distribution.D(wh) * m_distribution.G(wo, wi) * F /
        (4 * cosThetaO);
}

inline __host__ __device__
Spectrum GGXSmithReflectBSDF::F(
    const Vector3f& wo, 
    const Vector3f& wi, 
    Float* pdf) const
{
    Float cosThetaO = AbsCosTheta(wo), cosThetaI = AbsCosTheta(wi);
    Vector3f wh = wi + wo;
    // Handle degenerate cases for microfacet reflection
    if (cosThetaI == 0 || cosThetaO == 0) return Spectrum(0.);
    if (wh.x == 0 && wh.y == 0 && wh.z == 0) return Spectrum(0.);
    wh = Normalize(wh);
    Spectrum F;
    if (m_conductor) {
        F = FrConductor(abs(Dot(wi, wh)), m_conductorEtaI, m_conductorEtaT, m_conductorK);
    }
    else {
        F = FrDielectric(Dot(wi, wh), m_dielectricEtaI, m_dielectricEtaT);
    }
    *pdf = m_distribution.Pdf(wo, wh) / (4 * Dot(wo, wh));
    return m_r * m_distribution.D(wh) * m_distribution.G(wo, wi) * F /
        (4 * cosThetaO);
}

#endif // __BSDF_H