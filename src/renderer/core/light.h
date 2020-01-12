#pragma once
#ifndef __LIGHT_H
#define __LIGHT_H

#include "renderer/core/fwd.h"
#include "renderer/core/parameterset.h"
#include "renderer/core/spectrum.h"
#include "renderer/core/transform.h"

class Light {
public:
    enum LightType {
        AREA_LIGHT = 0,
        SPOT_LIGHT,
        INFINITE_LIGHT,
    };

    Light() {}
    Light(
        LightType type,
        const Spectrum& L,
        int shapeID);
    Light(
        LightType type,
        const Spectrum& L);
    Light(
        LightType type,
        const Transform& lightToWorld,
        const Spectrum& I,
        Float totalWidth,
        Float falloffStart);

    bool isDelta() const;


    /*
    Spectrum Le() const;
    Spectrum L(
        const Normal3f& n,
        const Vector3f& w) const;
    Spectrum Sample(
        const Interaction& inter,
        const Triangle* triangle,
        Vector3f* wi,
        Float* pdf,
        Ray* visibility,
        unsigned int& seed) const;
    Float Pdf(
        const Triangle* triangle,
        const Interaction& inter,
        const Vector3f& wi) const;

    Float Falloff(
        const Vector3f& w) const;
    */


    // Global
    LightType m_type;
    Transform m_lightToWorld;
    // Area Light
    int m_shapeID;
    // Area Light & Infinite Light
    Spectrum m_L;
    // Spot Light
    Point3f m_pLight;
    Spectrum m_I;
    Float m_cosTotalWidth, m_cosFalloffStart;
};

std::shared_ptr<Light>
CreateAreaLight(
    const ParameterSet& params,
    int shapeID);

std::shared_ptr<Light>
CreateSpotLight(
    const ParameterSet& params,
    const Transform& lightToWorld);

std::shared_ptr<Light>
CreateInfiniteLight(
    const ParameterSet& params);

inline __device__ __host__
bool Light::isDelta() const
{
    if (m_type == SPOT_LIGHT) {
        return true;
    }
    else {
        return false;
    }
}

#endif // !__LIGHT_H
