#pragma once

#include "renderer/ray.h"
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
        const Differential& geom,
        Material& material)
    : m_frame(geom), m_bxdf_num(0)
    { 
        if (material.m_type == MATERIAL_DIFFUSE)
        {
            addBxDF(CreateLambertReflectBxDF(geom, material));
        }
        else if (material.m_type == MATERIAL_SPECULAR)
        {
            addBxDF(CreateFresnelSpecularBxDF(geom, material));
        }
        else if (material.m_type == MATERIAL_ROUGH_CONDUCTOR)
        {
            addBxDF(CreateMicrofacetReflectBxDF(geom, material, true));
        }
        else if (material.m_type == MATERIAL_ROUGH_PLASTIC)
        {                        
            //addBxDF(CreateFresnelBlendBxDF(geom, material));
            addBxDF(CreateLambertReflectBxDF(geom, material));            
            Material mat(material);
            mat.m_color = material.m_color1;
            addBxDF(CreateMicrofacetReflectBxDF(geom, mat, false));            
        }
        else
        {
            assert(false);
        }
    }

    HOST_DEVICE
    void addBxDF(const BxDF& bxdf)
    {
        assert(m_bxdf_num < MAX_BXDF_NUM);
        m_bxdfs[m_bxdf_num++] = bxdf;
    }

    /**
     * \brief evaluate f = BSDF * cos
     * \param bsdf_record 
     */
    HOST_DEVICE
    void eval(
        BSDFSample& bsdf_record) const
    {
        bsdf_record.m_f = Spectrum(0.f);
        float3 local_wo = m_frame.worldToLocal(bsdf_record.m_wo);
        float3 local_wi = m_frame.worldToLocal(bsdf_record.m_wi);
        if (local_wo.z == 0) return;
        bool reflect = local_wo.z * local_wi.z > 0;

        for (uint32 i = 0; i < m_bxdf_num; i++)
        {
            Spectrum f;
            if ((reflect && (m_bxdfs[i].m_property & BXDF_REFLECTION)) ||
                (!reflect && (m_bxdfs[i].m_property & BXDF_TRANSMISSION)))
            {
                m_bxdfs[i].eval(local_wo, local_wi, &f);
            }
            bsdf_record.m_f += f;
        }

        bsdf_record.m_specular = isDelta();
    }

    /**
     * \brief evaluate pdf
     * \param bsdf_record 
     */
    HOST_DEVICE
    void pdf(
        BSDFSample&  bsdf_record) const
    {
        bsdf_record.m_pdf = 0.f;
        float3 local_wo = m_frame.worldToLocal(bsdf_record.m_wo);
        float3 local_wi = m_frame.worldToLocal(bsdf_record.m_wi);
        if (local_wo.z == 0) return;
        bool reflect = local_wo.z * local_wi.z > 0;

        for (uint32 i = 0; i < m_bxdf_num; i++)
        {
            float pdf;
            m_bxdfs[i].pdf(local_wo, local_wi, &pdf);
            bsdf_record.m_pdf += pdf;            
        }    
        bsdf_record.m_pdf / m_bxdf_num;

        bsdf_record.m_specular = isDelta();
    }

    /**
     * \brief sample wi, evaluate f and pdf
     * \param u random number
     * \param bsdf_record 
     */
    HOST_DEVICE
    void sample(
        const float2& u,
        BSDFSample&   bsdf_record) const
    {
        bsdf_record.m_f = Spectrum(0.f);
        bsdf_record.m_pdf = 0.f;
        float3 local_wo = m_frame.worldToLocal(bsdf_record.m_wo);
        float3 local_wi;

        float2 v = u;
        // Sample which bxdf
        uint32 idx = min((int)floor(v.x * m_bxdf_num), m_bxdf_num - 1);
        // Remap
        v.x = v.x * m_bxdf_num - idx;
        if (local_wo.z == 0) return;
        // Sample bxdf
        m_bxdfs[idx].sample(v, local_wo, &local_wi, &bsdf_record.m_f, &bsdf_record.m_pdf);
        bsdf_record.m_specular = m_bxdfs[idx].isDelta();
        // If sample fail then return
        if (bsdf_record.m_pdf == 0.f) return;        
        // Transform wi form local to world
        bsdf_record.m_wi = m_frame.localToWorld(local_wi);
        
        // Evaluate pdf
        if (!m_bxdfs[idx].isDelta())
        {
            for (uint32 i = 0; i < m_bxdf_num; i++)
            {
                if (i != idx)
                {
                    float pdf;
                    m_bxdfs[i].pdf(local_wo, local_wi, &pdf);
                    bsdf_record.m_pdf += pdf;
                }
            }
        }
        bsdf_record.m_pdf / m_bxdf_num;

        // Evaluate f
        if (!m_bxdfs[idx].isDelta())
        {
            bool reflect = local_wo.z * local_wi.z > 0;
            bsdf_record.m_f = Spectrum(0.f);
            for (uint32 i = 0; i < m_bxdf_num; i++)
            {
                Spectrum f;
                if ((reflect && (m_bxdfs[i].m_property & BXDF_REFLECTION)) ||
                    (!reflect && (m_bxdfs[i].m_property & BXDF_TRANSMISSION)))
                {
                    m_bxdfs[i].eval(local_wo, local_wi, &f);
                }
                bsdf_record.m_f += f;
            }
        }
    }

    /**
     * \brief for NEE and MIS
     * \return return false iff it exists one no-specular BxDF component
     */
    HOST_DEVICE
    bool isDelta() const 
    { 
        for (uint32 i = 0; i < m_bxdf_num; i++)
        {
            if (!m_bxdfs[i].isDelta())
            {
                return false;
            }
        }
        return true;
    }

private:
    Frame m_frame;

    BxDF m_bxdfs[MAX_BXDF_NUM];
    uint32 m_bxdf_num;
};