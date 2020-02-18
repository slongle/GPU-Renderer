#pragma once

#include "renderer/spectrum.h"
#include "renderer/texture.h"

enum MaterialType
{
    MATERIAL_DIFFUSE,
    MATERIAL_SPECULAR,
    MATERIAL_ROUGH_CONDUCTOR,
    MATERIAL_ROUGH_DIELECTRIC,    
};

class Material {
public:
    Material() {}

    HOST_DEVICE
    bool isEmission() const 
    {
        return !isBlack(m_emission);
    }

public:
    // Spectrum
    Spectrum m_emission;
    TextureView m_color;
    TextureView m_etaI, m_etaT, m_k;
    // Float
    TextureView m_ior;
    TextureView m_alpha_x, m_alpha_y;

    MaterialType m_type;
};