#pragma once

#include "renderer/fwd.h"
#include "renderer/material.h"

#include "renderer/bsdfs/lambert.h"
#include "renderer/bsdfs/specular.h"
#include "renderer/bsdfs/microfacet.h"

struct BSDFData
{
    
};

/*
enum BSDFProperty
{
    BSDF_REFLECTION = 1 << 0,
    BSDF_TRANSMISSION = 1 << 1,
    BSDF_DIFFUSE = 1 << 2,
    BSDF_GLOSSY = 1 << 3,
    BSDF_SPECULAR = 1 << 4,
};
*/

class BXDF
{
public:

};



enum BSDFType 
{
    BSDF_REFLECTION = 1 << 0,
    BSDF_TRANSMISSION = 1 << 1,
    BSDF_DIFFUSE = 1 << 2,
    BSDF_GLOSSY = 1 << 3,
    BSDF_SPECULAR = 1 << 4,
};

class BSDF
{
public:
    HOST_DEVICE
    BSDF() {}

    HOST_DEVICE
    BSDF(
        const float3& normal_g,
        const float3& normal_s,
        const Material& material)
    : m_normal_g(normal_g), m_normal_s(normal_s)
    { 
        // Build Local Coordinate
        const float3 n = normal_s;
        if (fabsf(n.x) > fabsf(n.y)) {
            float inv_len = 1.f / sqrtf(n.x * n.x + n.z * n.z);
            m_s = make_float3(n.z * inv_len, 0, -n.x * inv_len);
        }
        else {
            float inv_len = 1.f / sqrtf(n.y * n.y + n.z * n.z);
            m_s = make_float3(0, n.z * inv_len, -n.y * inv_len);
        }
        m_t = cross(m_s, n);

        // Initialize BSDF
        m_material_type = material.m_type;
        if (material.m_ior == 0)
        {
            m_lambert_reflect = LambertReflect(material.m_diffuse);
            m_type = BSDFType(BSDF_REFLECTION | BSDF_DIFFUSE);
        }
        else
        {
            //printf("%f\n", material.m_ior);
            //m_lambert_reflect = LambertReflect(material.m_diffuse);
            m_specular = FresnelSpecular(material.m_specular, material.m_specular, 1.f, material.m_ior);
            m_type = BSDFType(BSDF_REFLECTION | BSDF_TRANSMISSION | BSDF_SPECULAR);
        }
    }

    /// evaluate eval = BSDF * cos
    ///
    HOST_DEVICE
    Spectrum eval(
        BSDFSample& bsdf_record) const
    {
        float3 local_wo = worldToLocal(bsdf_record.m_wo);
        float3 local_wi = worldToLocal(bsdf_record.m_wi);
        bool reflect = local_wo.z * local_wi.z > 0;        
        if ((m_type & BSDF_SPECULAR) != 0)
        {
            return m_specular.eval(local_wo, local_wi);
        }
        else
        {
            return reflect ? m_lambert_reflect.eval(local_wo, local_wi) : Spectrum(0.f);
        }        
    }

    /// evaluate pdf
    ///
    HOST_DEVICE
    void pdf(
        BSDFSample&  bsdf_record) const
    {
        float3 local_wo = worldToLocal(bsdf_record.m_wo);
        float3 local_wi = worldToLocal(bsdf_record.m_wi);
        bool reflect = local_wo.z * local_wi.z > 0;
        if ((m_type & BSDF_SPECULAR) != 0)
        {
            m_specular.pdf(local_wo, local_wi, &bsdf_record.m_pdf);
        }
        else
        {
            if (reflect)
            {
                m_lambert_reflect.pdf(local_wo, local_wi, &bsdf_record.m_pdf);
            }
            else
            {
                bsdf_record.m_pdf = 0;
            }
        }        
    }

    /// sample wi, evaluate f and pdf
    ///
    HOST_DEVICE
    void sample(
        BSDFSample&   bsdf_record,
        const float2  s) const
    {
        float3 local_wo = worldToLocal(bsdf_record.m_wo);
        float3 local_wi;
        
        if ((m_type & BSDF_SPECULAR) != 0)
        {
            m_specular.sample(s, local_wo, &local_wi, &bsdf_record.m_f, &bsdf_record.m_pdf);
        }
        else 
        {
            m_lambert_reflect.sample(s, local_wo, &local_wi, &bsdf_record.m_f, &bsdf_record.m_pdf);
        }
        
        bsdf_record.m_wi = localToWorld(local_wi);
    }

    HOST_DEVICE
    bool isDelta() const { return (m_type & BSDF_SPECULAR) != 0; }

//private:

    HOST_DEVICE
    float3 localToWorld(
        const float3& v) const
    {
        return normalize(m_s * v.x + m_t * v.y + m_normal_s * v.z);
    }

    HOST_DEVICE
    float3 worldToLocal(
        const float3& v) const 
    {
        return normalize(make_float3(dot(v, m_s), dot(v, m_t), dot(v, m_normal_s)));
    }

    float3 m_normal_g;
    float3 m_normal_s;
    float3 m_s;
    float3 m_t;

    BSDFType m_type;
    MaterialType m_material_type;

    LambertReflect m_lambert_reflect;
    FresnelSpecular m_specular;
    MicrofacetConductor m_conductor;
};