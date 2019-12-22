#pragma once
#ifndef __MATERIAL_H
#define __MATERIAL_H

#include "renderer/core/fwd.h"
#include "renderer/core/parameterset.h"
#include "renderer/core/spectrum.h"
#include "renderer/core/interaction.h"
#include "renderer/core/sampling.h"
#include "renderer/core/bsdf.h"

class Material {
public:
    enum ComponentType {
        DIFFUSE_REFLECT = 0x1u, 
        GLOSSY_REFLECT  = 0x2u,
    };

    // Diffuse Material
    Material(
        ComponentType type,
        Spectrum Kd);
    // Metal Material
    Material(
        ComponentType type,
        const Spectrum& eta,
        const Spectrum& k,
        const Float& uroughness,
        const Float& vroughness);
   
    Spectrum Sample(const Normal3f& n, const Vector3f& worldWo, Vector3f* worldWi, Float* pdf, unsigned int& seed) const;
    Spectrum F(const Normal3f& n, const Vector3f& worldWo, const Vector3f& worldWi) const;
    Spectrum F(const Normal3f& n, const Vector3f& worldWo, const Vector3f& worldWi, Float* pdf) const;    

    // Global
    ComponentType m_type;

    LambertReflectBSDF m_diffuseReflect;
    GGXSmithReflectBSDF m_glossyReflect;
};

std::shared_ptr<Material>
CreateMatteMaterial(
    const ParameterSet& param);

std::shared_ptr<Material>
CreateMetalMaterial(
    const ParameterSet& param);

inline __device__ __host__
Spectrum Material::F(
    const Normal3f& n,
    const Vector3f& worldWo,
    const Vector3f& worldWi,
    Float* pdf) const
{
    Vector3f s, t;
    CoordinateSystem(n, &s, &t);
    Vector3f localWo = WorldToLocal(worldWo, n, s, t);
    Vector3f localWi = WorldToLocal(worldWi, n, s, t);
    Spectrum cosBSDF(0);

    if (m_type == DIFFUSE_REFLECT) {
        cosBSDF = m_diffuseReflect.F(localWo, localWi, pdf);
    }
    else if (m_type == GLOSSY_REFLECT) {
        cosBSDF = m_glossyReflect.F(localWo, localWi, pdf);
    }

    return cosBSDF;
}

inline __device__ __host__
Spectrum Material::F(
    const Normal3f& n,
    const Vector3f& worldWo,
    const Vector3f& worldWi) const
{
    Vector3f s, t;
    CoordinateSystem(n, &s, &t);
    Vector3f localWo = WorldToLocal(worldWo, n, s, t);
    Vector3f localWi = WorldToLocal(worldWi, n, s, t);
    Spectrum cosBSDF(0);

    if (m_type == DIFFUSE_REFLECT) {
        cosBSDF = m_diffuseReflect.F(localWo, localWi);
    }
    else if (m_type == GLOSSY_REFLECT) {
        cosBSDF = m_glossyReflect.F(localWo, localWi);
    }

    return cosBSDF;
}
inline __device__ __host__
Spectrum Material::Sample(
    const Normal3f& n,
    const Vector3f& worldWo,
    Vector3f* worldWi,
    Float* pdf,
    unsigned int& seed) const
{
    Vector3f s, t;
    CoordinateSystem(n, &s, &t);
    Vector3f localWo = WorldToLocal(worldWo, n, s, t);
    Vector3f localWi;
    Spectrum cosBSDF(0);    

    if (m_type == DIFFUSE_REFLECT) {
        cosBSDF = m_diffuseReflect.Sample(localWo, &localWi, pdf, seed);
    }
    else if (m_type == GLOSSY_REFLECT) {   
        //printf("!");
        cosBSDF = m_glossyReflect.Sample(localWo, &localWi, pdf, seed);
    }

    //return cosBSDF;
    *worldWi = LocalToWorld(localWi, n, s, t);
    return cosBSDF;
}

#endif // !__MATERIAL_H
