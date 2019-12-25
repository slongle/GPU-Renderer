#pragma once
#ifndef __BSDF_H
#define __BSDF_H

#include "renderer/core/spectrum.h"
#include "renderer/core/interaction.h"
#include "renderer/core/microfacet.h"
#include "renderer/core/optic.h"

class LambertReflectBSDF {
public:
    LambertReflectBSDF() {}
    LambertReflectBSDF(
        const Spectrum& r);

    Spectrum Sample(const Vector3f& wo, Vector3f* wi, Float* pdf, unsigned int& seed) const;
    Spectrum F(const Vector3f& wo, const Vector3f& wi) const;
    Spectrum F(const Vector3f& wo, const Vector3f& wi, Float* pdf) const;

    Spectrum m_r;
};

class SpecularReflectBSDF {
public:
    SpecularReflectBSDF() {}

    // Conductor
    SpecularReflectBSDF(
        const Spectrum& r,
        const bool& conductor,
        const Spectrum& etaI,
        const Spectrum& etaT,
        const Spectrum& k);

    // Dielectric
    SpecularReflectBSDF(
        const Spectrum& r,
        const bool& conductor,
        const Float& etaI,
        const Float& etaT);

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
};

class SpecularTransmission {
public:
    SpecularTransmission() {}
    SpecularTransmission(
        const Spectrum& t,
        const Float& etaA,
        const Float& etaB);

    Spectrum Sample(const Vector3f& wo, Vector3f* wi, Float* pdf, unsigned int& seed) const;
    Spectrum F(const Vector3f& wo, const Vector3f& wi) const;
    Spectrum F(const Vector3f& wo, const Vector3f& wi, Float* pdf) const;

    // Global
    Spectrum m_t;
    // Dielectric Fresnel
    Float m_etaA, m_etaB;
};

class FresnelSpecular {
public:
    FresnelSpecular() {}
    FresnelSpecular(
        const Spectrum& t,
        const Spectrum& r,
        const Float& etaA,
        const Float& etaB);

    Spectrum Sample(const Vector3f& wo, Vector3f* wi, Float* pdf, unsigned int& seed) const;
    Spectrum F(const Vector3f& wo, const Vector3f& wi) const;
    Spectrum F(const Vector3f& wo, const Vector3f& wi, Float* pdf) const;

    // Global
    Spectrum m_t, m_r;
    // Dielectric Fresnel
    Float m_etaA, m_etaB;
};

class GGXSmithReflectBSDF {
public:
    GGXSmithReflectBSDF() {}

    // Conductor
    GGXSmithReflectBSDF(
        const Spectrum& r, 
        const bool& conductor,
        const Spectrum& etaI, 
        const Spectrum& etaT,
        const Spectrum& k,
        const Float& uroughness,
        const Float& vroughness);

    // Dielectric
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


class GGXSmithTransmission {
public:
    GGXSmithTransmission() {}

    GGXSmithTransmission(
        const Spectrum& t, 
        const Float& etaA, 
        const Float& etaB,
        const Float& uroughness,
        const Float& vroughness);

    Spectrum Sample(const Vector3f& wo, Vector3f* wi, Float* pdf, unsigned int& seed) const;
    Spectrum F(const Vector3f& wo, const Vector3f& wi) const;
    Spectrum F(const Vector3f& wo, const Vector3f& wi, Float* pdf) const;
    Float Pdf(const Vector3f& wo, const Vector3f& wi) const;

    // Global
    Spectrum m_t;
    // Dielectric Fresnel
    Float m_etaA, m_etaB;
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
    if (wo.z < 0) wi->z *= -1;
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
    *pdf = SameHemisphere(wo, wi) ? CosineSampleHemispherePdf(AbsCosTheta(wi)) : 0;
    return m_r * InvPi * AbsCosTheta(wi);
}

inline
SpecularReflectBSDF::SpecularReflectBSDF(
    const Spectrum& r,
    const bool& conductor,
    const Spectrum& etaI,
    const Spectrum& etaT,
    const Spectrum& k)
    : m_r(r), m_conductor(conductor), m_conductorEtaI(etaI), m_conductorEtaT(etaT), m_conductorK(k)
{
}

inline
SpecularReflectBSDF::SpecularReflectBSDF(
    const Spectrum& r,
    const bool& conductor,
    const Float& etaI,
    const Float& etaT)
    : m_r(r), m_conductor(conductor), m_dielectricEtaI(etaI), m_dielectricEtaT(etaT)
{
}

inline __host__ __device__
Spectrum SpecularReflectBSDF::Sample(
    const Vector3f& wo, 
    Vector3f* wi, 
    Float* pdf, 
    unsigned int& seed) const
{
    *wi = Vector3f(-wo.x, -wo.y, wo.z);
    *pdf = 1;
    Spectrum F;
    if (m_conductor) {
        F = FrConductor(AbsCosTheta(*wi), m_conductorEtaI, m_conductorEtaT, m_conductorK);
    }
    else {
        F = FrDielectric(CosTheta(*wi), m_dielectricEtaI, m_dielectricEtaT);
    }
    return F * m_r / AbsCosTheta(*wi);
}

inline __host__ __device__
Spectrum SpecularReflectBSDF::F(
    const Vector3f& wo, 
    const Vector3f& wi) const
{
    return Spectrum(0);
}

inline __host__ __device__
Spectrum SpecularReflectBSDF::F(
    const Vector3f& wo, 
    const Vector3f& wi, 
    Float* pdf) const
{
    *pdf = 0;
    return Spectrum(0);
}

inline
SpecularTransmission::SpecularTransmission(
    const Spectrum& t,
    const Float& etaA,
    const Float& etaB)
    : m_t(t), m_etaA(etaA), m_etaB(etaB)
{
}

inline __host__ __device__
Spectrum SpecularTransmission::Sample(
    const Vector3f& wo, 
    Vector3f* wi, 
    Float* pdf, 
    unsigned int& seed) const
{
    // Figure out which $\eta$ is incident and which is transmitted
    bool entering = CosTheta(wo) > 0;
    Float etaI = entering ? m_etaA : m_etaB;
    Float etaT = entering ? m_etaB : m_etaA;

    // Compute ray direction for specular transmission
    if (!Refract(wo, Faceforward(Normal3f(0, 0, 1), wo), etaI / etaT, wi))
        return 0;
    *pdf = 1;
    Spectrum ft = m_t * (Spectrum(1.) - FrDielectric(CosTheta(*wi), m_etaA, m_etaB));
    // Account for non-symmetry with transmission to different medium
    //if (mode == TransportMode::Radiance) 
    ft *= (etaI * etaI) / (etaT * etaT);
    return ft / AbsCosTheta(*wi);
}

inline __host__ __device__
Spectrum SpecularTransmission::F(
    const Vector3f& wo, 
    const Vector3f& wi) const
{
    return Spectrum(0);
}

inline __host__ __device__
Spectrum SpecularTransmission::F(
    const Vector3f& wo, 
    const Vector3f& wi, 
    Float* pdf) const
{
    *pdf = 0;
    return Spectrum(0);
}

inline
FresnelSpecular::FresnelSpecular(
    const Spectrum& t,
    const Spectrum& r,
    const Float& etaA,
    const Float& etaB)
    : m_t(t), m_r(r), m_etaA(etaA), m_etaB(etaB) 
{
}

inline __host__ __device__
Spectrum FresnelSpecular::Sample(
    const Vector3f& wo, 
    Vector3f* wi, 
    Float* pdf, 
    unsigned int& seed) const
{
    Float F = FrDielectric(CosTheta(wo), m_etaA, m_etaB);
    Float u = NextRandom(seed);
    if (u < F) {
        // Compute specular reflection for _FresnelSpecular_

        // Compute perfect specular reflection direction
        *wi = Vector3f(-wo.x, -wo.y, wo.z);
        
        *pdf = F;
        return m_r * F;
    }
    else {
        // Compute specular transmission for _FresnelSpecular_

        // Figure out which $\eta$ is incident and which is transmitted
        bool entering = CosTheta(wo) > 0;
        Float etaI = entering ? m_etaA : m_etaB;
        Float etaT = entering ? m_etaB : m_etaA;

        // Compute ray direction for specular transmission        
        if (!Refract(wo, Faceforward(Normal3f(0, 0, 1), wo), etaI / etaT, wi))
            return Spectrum(0.);
        Spectrum ft = m_t * (1 - F);

        // Account for non-symmetry with transmission to different medium
        //if (mode == TransportMode::Radiance)
        ft *= (etaI * etaI) / (etaT * etaT);

        *pdf = 1 - F;
        return ft;
    }
}

inline __host__ __device__
Spectrum FresnelSpecular::F(
    const Vector3f& wo, 
    const Vector3f& wi) const
{
    return Spectrum(0.f);
}

inline __host__ __device__
Spectrum FresnelSpecular::F(
    const Vector3f& wo, 
    const Vector3f& wi, 
    Float* pdf) const
{
    *pdf = 0.f;
    return Spectrum(0.f);
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

inline __host__ __device__
GGXSmithTransmission::GGXSmithTransmission(
    const Spectrum& t,
    const Float& etaA,
    const Float& etaB,
    const Float& uroughness,
    const Float& vroughness)
    : m_t(t), m_etaA(etaA), m_etaB(etaB), m_distribution(uroughness, vroughness)
{
}

inline __host__ __device__
Spectrum GGXSmithTransmission::Sample(
    const Vector3f& wo,
    Vector3f* wi, 
    Float* pdf, 
    unsigned int& seed) const
{
    if (wo.z == 0) return 0.;
    Vector3f wh = m_distribution.Sample_wh(wo, seed);
    Float eta;
    if (CosTheta(wo) > 0) {
        eta = (m_etaA / m_etaB);
    }
    else {
        eta = (m_etaB / m_etaA);
    }
    if (!Refract(wo, (Normal3f)wh, eta, wi)) return Spectrum(0);    
    *pdf = Pdf(wo, *wi);
    return F(wo, *wi);
}

inline __host__ __device__
Spectrum GGXSmithTransmission::F(
    const Vector3f& wo, 
    const Vector3f& wi) const
{
    if (SameHemisphere(wo, wi)) Spectrum(0);  // transmission only

    Float cosThetaO = CosTheta(wo);
    Float cosThetaI = CosTheta(wi);
    if (cosThetaI == 0 || cosThetaO == 0) return Spectrum(0);

    // Compute $\wh$ from $\wo$ and $\wi$ for microfacet transmission
    Float eta = CosTheta(wo) > 0 ? (m_etaB / m_etaA) : (m_etaA / m_etaB);
    Vector3f wh = Normalize(wo + wi * eta);
    if (wh.z < 0) wh = -wh;

    Spectrum F = FrDielectric(Dot(wo, wh), m_etaA, m_etaB);

    Float sqrtDenom = Dot(wo, wh) + eta * Dot(wi, wh);
    Float factor = 1 / eta;

    return (Spectrum(1.f) - F) * m_t *
        abs(m_distribution.D(wh) * m_distribution.G(wo, wi) * eta * eta *
            AbsDot(wi, wh) * AbsDot(wo, wh) * factor * factor /
            (cosThetaO * sqrtDenom * sqrtDenom));    
}

inline __host__ __device__
Spectrum GGXSmithTransmission::F(
    const Vector3f& wo, 
    const Vector3f& wi, 
    Float* pdf) const
{
    if (SameHemisphere(wo, wi)) Spectrum(0);  // transmission only

    Float cosThetaO = CosTheta(wo);
    Float cosThetaI = CosTheta(wi);
    if (cosThetaI == 0 || cosThetaO == 0) return Spectrum(0);

    // Compute $\wh$ from $\wo$ and $\wi$ for microfacet transmission
    Float eta = CosTheta(wo) > 0 ? (m_etaB / m_etaA) : (m_etaA / m_etaB);
    Vector3f wh = Normalize(wo + wi * eta);
    if (wh.z < 0) wh = -wh;

    Spectrum F = FrDielectric(Dot(wo, wh), m_etaA, m_etaB);

    Float sqrtDenom = Dot(wo, wh) + eta * Dot(wi, wh);
    Float factor = 1 / eta;

    *pdf = Pdf(wo, wi);

    return (Spectrum(1.f) - F) * m_t *
        abs(m_distribution.D(wh) * m_distribution.G(wo, wi) * eta * eta *
            AbsDot(wi, wh) * AbsDot(wo, wh) * factor * factor /
            (cosThetaO * sqrtDenom * sqrtDenom));
}

inline __host__ __device__
Float GGXSmithTransmission::Pdf(
    const Vector3f& wo, 
    const Vector3f& wi) const
{
    if (SameHemisphere(wo, wi)) return 0;
    // Compute $\wh$ from $\wo$ and $\wi$ for microfacet transmission
    Float eta = CosTheta(wo) > 0 ? (m_etaB / m_etaA) : (m_etaA / m_etaB);
    Vector3f wh = Normalize(wo + wi * eta);

    // Compute change of variables _dwh\_dwi_ for microfacet transmission
    Float sqrtDenom = Dot(wo, wh) + eta * Dot(wi, wh);
    Float dwh_dwi = abs((eta * eta * Dot(wi, wh)) / (sqrtDenom * sqrtDenom));
    return m_distribution.Pdf(wo, wh) * dwh_dwi;
}

#endif // __BSDF_H