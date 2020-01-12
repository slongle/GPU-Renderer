#include "light.h"

#include "renderer/core/sampling.h"

Light::Light(
    LightType type,
    const Spectrum& L,
    int shapeID)
    : m_type(type), m_L(L), m_shapeID(shapeID)
{
}

Light::Light(
    LightType type,
    const Spectrum& L)
    : m_type(type), m_L(L)
{
}

Light::Light(
    LightType type,
    const Transform& lightToWorld,
    const Spectrum& I,
    Float totalWidth,
    Float falloffStart)
    : m_type(type), m_lightToWorld(lightToWorld), m_I(I),
    m_cosTotalWidth(cos(Radians(totalWidth))),
    m_cosFalloffStart(cos(Radians(falloffStart)))
{
}

std::shared_ptr<Light>
CreateAreaLight(
    const ParameterSet& params,
    int shapeID)
{
    std::vector<Float> rgbs = params.GetSpectrum("L");
    Spectrum L(rgbs);
    return std::make_shared<Light>(Light::AREA_LIGHT, L, shapeID);
}

std::shared_ptr<Light> CreateSpotLight(
    const ParameterSet& params,
    const Transform& lightToWorld)
{
    Spectrum I(params.GetSpectrum("I", std::vector<Float>{1.f, 1.f, 1.f}));
    Spectrum scale(params.GetSpectrum("scale", std::vector<Float>{1.f, 1.f, 1.f}));
    Float coneangle = params.GetFloat("coneangle", 30.f);
    Float conedelta = params.GetFloat("conedeltaangle", 5.f);
    // Compute spotlight world to light transformation
    Point3f from = params.GetPoint("from", Point3f(0, 0, 0));
    Point3f to = params.GetPoint("to", Point3f(0, 0, 1));
    Vector3f dir = Normalize(to - from);
    Vector3f du, dv;
    CoordinateSystem(Normal3f(dir), &du, &dv);
    Transform dirToZ =
        Transform(Matrix4x4(du.x, du.y, du.z, 0., dv.x, dv.y, dv.z, 0., dir.x,
            dir.y, dir.z, 0., 0, 0, 0, 1.));
    Transform light2world =
        lightToWorld * Translate(from.x, from.y, from.z) * Inverse(dirToZ);
    return std::make_shared<Light>(Light::SPOT_LIGHT, light2world, I * scale, coneangle, coneangle - conedelta);
}

std::shared_ptr<Light> CreateInfiniteLight(
    const ParameterSet& params)
{
    Spectrum L(params.GetSpectrum("L", std::vector<Float>{1.f, 1.f, 1.f}));
    return std::make_shared<Light>(Light::INFINITE_LIGHT, L);
}

