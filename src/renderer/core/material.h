#pragma once
#ifndef __MATERIAL_H
#define __MATERIAL_H

#include "renderer/core/fwd.h"
#include "renderer/core/parameterset.h"
#include "renderer/core/spectrum.h"
#include "renderer/core/interaction.h"
#include "renderer/core/sampling.h"

class Material {
public:
    enum MaterialType {
        DIFFUSE_MATERIAL = 0
    };

    Material(MaterialType type, Spectrum Kd);

    __device__ __host__ Spectrum Sample(Interaction& interaction, unsigned int& seed) const;


    __device__ __host__ void BuildLocalCoordinate(const Interaction& interaction);
    __device__ __host__ Vector3f WorldToLocal(const Vector3f& v) const;
    __device__ __host__ Vector3f LocalToWorld(const Vector3f& v) const;

    // Global
    MaterialType m_type;

    // Diffuse
    Spectrum m_Kd;
};

std::shared_ptr<Material>
CreateMatteMaterial(
    const ParameterSet& param);

/*
 * return fr * |cos|
 */
inline __device__ __host__ 
Spectrum Material::Sample(
    Interaction& interaction, 
    unsigned int& seed) const
{
    Spectrum ret(1);
    Vector3f wo = WorldToLocal(interaction.m_wo), wi;
    if (m_type == DIFFUSE_MATERIAL) {
         wi = SampleCosineHemisphere(seed);
         //PI;
         //ret *= m_Kd * INV_PI * (AbsCos(wi) / );
    }
    interaction.m_wi = LocalToWorld(wi);
    return ret;
}

inline __device__ __host__ 
void Material::BuildLocalCoordinate(const Interaction& interaction)
{
    
}

inline __device__ __host__ 
Vector3f Material::WorldToLocal(const Vector3f& v) const
{
    return Vector3f();
}

inline __device__ __host__ 
Vector3f Material::LocalToWorld(const Vector3f& v) const
{
    return Vector3f();
}


#endif // !__MATERIAL_H
