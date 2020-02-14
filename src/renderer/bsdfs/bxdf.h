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

    /**
     * \brief evaluate f = BSDF * cos
     * \param wo      out direction in local coordinate
     * \param wi incident direction in local coordinate
     * \param f 
     */
    HOST_DEVICE
    void eval(const float3& wo, const float3& wi, Spectrum* f) const;

    /**
     * \brief evaluate pdf
     * \param wo      out direction in local coordinate
     * \param wi incident direction in local coordinate
     * \param pdf probability of (wo, wi)
     */
    HOST_DEVICE
    void pdf(const float3& wo, const float3& wi, float* pdf) const;
    
    /**
     * \brief sample wi, evaluate f and pdf
     * \param u random number
     * \param wo      out direction in local coordinate
     * \param wi incident direction in local coordinate
     * \param f 
     * \param pdf probability of (wo, wi)
     */
    HOST_DEVICE
    void sample(const float2& u, const float3& wo, float3* wi, Spectrum* f, float* pdf) const;
    HOST_DEVICE
    bool isDelta() const { return (m_property & BXDF_SPECULAR) != 0; }

    Spectrum m_color;
    float m_ior;
    float m_alpha_x, m_alpha_y;
    Fresnel m_fresnel;
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
    else if (m_type == BXDF_MICROFACET_REFLECT)
    {
        MicrofacetReflectEval(wo, wi, f, m_color, m_alpha_x, m_alpha_y, m_fresnel);        
    }
    else
    {
        assert(false);
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
    else if (m_type == BXDF_MICROFACET_REFLECT)
    {
        MicrofacetReflectPdf(wo, wi, pdf, m_color, m_alpha_x, m_alpha_y);        
    }
    else
    {
        assert(false);
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
    else if (m_type == BXDF_MICROFACET_REFLECT)
    {
        MicrofacetReflectSample(u, wo, wi, f, pdf, m_color, m_alpha_x, m_alpha_y, m_fresnel);        
    }
    else
    {
        assert(false);
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

inline HOST_DEVICE
BxDF CreateMicrofacetReflectBxDF(const Material& material, bool conduct)
{
    BxDF ret;
    ret.m_color = material.m_color;
    ret.m_alpha_x = material.m_alpha_x;
    ret.m_alpha_y = material.m_alpha_y;    
    ret.m_fresnel = conduct ? Fresnel(material.m_etaI, material.m_etaT, material.m_k)
                            : Fresnel(material.m_etaI, material.m_etaT);
    ret.m_property = BxDFProperty(BXDF_REFLECTION | BXDF_GLOSSY);
    ret.m_type = BXDF_MICROFACET_REFLECT;
    return ret;
}