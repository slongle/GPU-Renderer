#pragma once

#include "renderer/material.h"

#include "renderer/bsdfs/bxdf.h"

#define MAX_BXDF_NUM 10

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
    : m_frame(normal_g, normal_s), m_bxdf_num(0)
    { 
        if (material.m_type == MATERIAL_DIFFUSE)
        {
            addBxDF(CreateLambertReflectBxDF(material));
        }
        else if (material.m_type == MATERIAL_SPECULAR)
        {
            addBxDF(CreateFresnelSpecularBxDF(material));
        }
    }

    HOST_DEVICE
    void addBxDF(const BxDF& bxdf)
    {
        assert(m_bxdf_num < MAX_BXDF_NUM);
        m_bxdfs[m_bxdf_num++] = bxdf;
    }

    /// evaluate eval = BSDF * cos
    ///
    HOST_DEVICE
    void eval(
        BSDFSample& bsdf_record) const
    {
        float3 local_wo = m_frame.worldToLocal(bsdf_record.m_wo);
        float3 local_wi = m_frame.worldToLocal(bsdf_record.m_wi);
        bool reflect = local_wo.z * local_wi.z > 0;    

        m_bxdfs[0].eval(local_wo, local_wi, &bsdf_record.m_f);
    }

    /// evaluate pdf
    ///
    HOST_DEVICE
    void pdf(
        BSDFSample&  bsdf_record) const
    {
        float3 local_wo = m_frame.worldToLocal(bsdf_record.m_wo);
        float3 local_wi = m_frame.worldToLocal(bsdf_record.m_wi);
        bool reflect = local_wo.z * local_wi.z > 0;

        m_bxdfs[0].pdf(local_wo, local_wi, &bsdf_record.m_pdf);
    }

    /// sample wi, evaluate f and pdf
    ///
    HOST_DEVICE
    void sample(
        const float2& u,
        BSDFSample&   bsdf_record) const
    {
        float3 local_wo = m_frame.worldToLocal(bsdf_record.m_wo);
        float3 local_wi;
        
        m_bxdfs[0].sample(u, local_wo, &local_wi, &bsdf_record.m_f, &bsdf_record.m_pdf);
        bsdf_record.m_specular = m_bxdfs[0].isDelta();

        bsdf_record.m_wi = m_frame.localToWorld(local_wi);
    }

    HOST_DEVICE
    bool isDelta() const 
    { 
        for (uint32 i = 0; i < m_bxdf_num; i++)
        {
            if (m_bxdfs[i].isDelta())
            {
                return true;
            }
        }
        return false;
    }

//private:
    Frame m_frame;

    BxDF m_bxdfs[MAX_BXDF_NUM];
    uint32 m_bxdf_num;
};