#pragma once

#include "spectrum.h"

enum MaterialType
{
    MATERIAL_DIFFUSE,
    MATERIAL_MIRROR,
    MATERIAL_GLASS,
};

class Material {
public:
    Material() {}
    Material(
        const Spectrum& diffuse, 
        const Spectrum& specular, 
        const Spectrum& emission,
        const float&    ior) :
        m_diffuse(diffuse), m_specular(specular), m_emission(emission), m_ior(ior) {}

    HOST_DEVICE
    bool isEmission() const 
    {
        return !isBlack(m_emission);
    }

    void setZero()
    {
        m_diffuse  = make_float3(0.f);
        m_specular = make_float3(0.f);
        m_emission = make_float3(0.f);
        m_ior = 0.f;
    }

    Spectrum m_diffuse;
    Spectrum m_specular;
    Spectrum m_emission;
    float    m_ior;

    MaterialType m_type;
};