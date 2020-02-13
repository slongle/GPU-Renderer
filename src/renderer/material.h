#pragma once

#include "spectrum.h"

enum MaterialType
{
    MATERIAL_DIFFUSE,
    MATERIAL_SPECULAR,
};

class Material {
public:
    Material() {}
    Material(
        const Spectrum& color, 
        const Spectrum& emission,
        const float&    ior) :
        m_color(color), m_emission(emission), m_ior(ior) {}

    HOST_DEVICE
    bool isEmission() const 
    {
        return !isBlack(m_emission);
    }

    void setZero()
    {
        m_color    = Spectrum(0.f);
        m_emission = Spectrum(0.f);
        m_ior = 0.f;
    }

    Spectrum m_color;
    Spectrum m_emission;
    float    m_ior;

    MaterialType m_type;
};