#pragma once

#include "renderer/material.h"
#include "renderer/bsdfs/lambert.h"
#include "renderer/bsdfs/specular.h"
#include "renderer/bsdfs/microfacet.h"

enum BxDFProperty
{
    BXDF_REFLECTION = 1 << 0,
    BXDF_TRANSMISSION = 1 << 1,
    BXDF_DIFFUSE = 1 << 2,
    BXDF_GLOSSY = 1 << 3,
    BXDF_SPECULAR = 1 << 4,
};

enum BxDFType
{
    BXDF_LAMBERT_REFLECT,
    BXDF_FRESNEL_SPECULAR,
    BXDF_MICROFACET_REFLECT,
    BXDF_MICROFACET_REFRACT,
};

class BxDF
{
public:

    HOST_DEVICE
    BxDF() {}

    // evaluate f = BSDF * cos, BSDF = color / PI
    void eval(const float3& wo, const float3& wi, Spectrum* f) const;

    // evaluate pdf
    void pdf(const float3& wo, const float3& wi, float* pdf) const;

    // sample wi, evaluate f and pdf
    void sample(const float2& u, const float3& wo, float3* wi, Spectrum* f, float* pdf) const;

    bool isDelta() const { return (m_property & BXDF_SPECULAR) != 0; }

    Spectrum m_color;
    float m_ior;
    BxDFProperty m_property;
    BxDFType m_type;
};

inline HOST_DEVICE
void BxDF::eval(
    const float3& wo, 
    const float3& wi, 
    Spectrum* f) const
{
    if (m_type == BXDF_LAMBERT_REFLECT)
    {
        LambertReflectEval(wo, wi, f, m_color);
    }
    else if (m_type == BXDF_FRESNEL_SPECULAR)
    {
        FresnelSpecularEval(wo, wi, f, m_color, m_ior);
    }
}

inline HOST_DEVICE
void BxDF::pdf(
    const float3& wo, 
    const float3& wi, 
    float* pdf) const
{
    if (m_type == BXDF_LAMBERT_REFLECT)
    {
        LambertReflectPdf(wo, wi, pdf, m_color);
    }
    else if (m_type == BXDF_FRESNEL_SPECULAR)
    {
        FresnelSpecularPdf(wo, wi, pdf, m_color, m_ior);
    }
}

inline HOST_DEVICE
void BxDF::sample(
    const float2& u,
    const float3& wo, 
    float3* wi, 
    Spectrum* f, 
    float* pdf) const
{
    if (m_type == BXDF_LAMBERT_REFLECT)
    {
        LambertReflectSample(u, wo, wi, f, pdf, m_color);
    }
    else if (m_type == BXDF_FRESNEL_SPECULAR)
    {
        FresnelSpecularSample(u, wo, wi, f, pdf, m_color, m_ior);
    }
}

inline HOST_DEVICE
BxDF CreateLambertReflectBxDF(const Material& material)
{
    BxDF ret;
    ret.m_color = material.m_color;
    ret.m_property = BxDFProperty(BXDF_REFLECTION | BXDF_DIFFUSE);
    ret.m_type = BXDF_LAMBERT_REFLECT;
    return ret;
}

inline HOST_DEVICE
BxDF CreateFresnelSpecularBxDF(const Material& material)
{
    BxDF ret;
    ret.m_color = material.m_color;
    ret.m_ior = material.m_ior;
    ret.m_property = BxDFProperty(BXDF_REFLECTION | BXDF_TRANSMISSION | BXDF_SPECULAR);
    ret.m_type = BXDF_FRESNEL_SPECULAR;
    return ret;
}