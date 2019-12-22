#include "material.h"

Material::Material(
    int type,
    Spectrum Kd)
    : m_type(type), m_diffuseReflect(Kd)
{
}

Material::Material(
    int type,
    const Spectrum& eta,
    const Spectrum& k,
    const Float& uroughness,
    const Float& vroughness)
    : m_type(type), m_glossyReflect(Spectrum(1), true, Spectrum(1), eta, k, uroughness, vroughness)
{
}

Material::Material(
    int type,
    const Spectrum& Kr,
    const Spectrum& Kt,
    const Float& eta,
    const Float& uroughness,
    const Float& vroughness)
    : m_type(type),
    m_glossyReflect(Kr, false, 1.f, eta, uroughness, vroughness),
    m_glossyTransmission(Kt, 1.f, eta, uroughness, vroughness)
{
}

std::shared_ptr<Material>
CreateMatteMaterial(
    const ParameterSet& param)
{
    std::vector<Float> rgbs = param.GetSpectrum("Kd");
    Spectrum Kd(rgbs);
    return std::make_shared<Material>(Material::DIFFUSE_REFLECT, Kd);
}

std::shared_ptr<Material> 
CreateMetalMaterial(
    const ParameterSet& param)
{
    Spectrum eta(param.GetSpectrum("eta"));
    Spectrum k(param.GetSpectrum("k"));
    //Float roughness = param.GetFloat("roughness", 0.01f);
    Float uroughness = param.GetFloat("uroughness");
    Float vroughness = param.GetFloat("vroughness");
    return std::make_shared<Material>(Material::GLOSSY_REFLECT, eta, k, uroughness, vroughness);
}

std::shared_ptr<Material> 
CreateGlassMaterial(
    const ParameterSet& param)
{
    Spectrum Kr(param.GetSpectrum("Kr", std::vector<Float>{1, 1, 1}));
    Spectrum Kt(param.GetSpectrum("Kt", std::vector<Float>{1, 1, 1}));
    Float eta = param.GetFloat("index", 1.5f);
    Float uroughness = param.GetFloat("uroughness", 0.f);
    Float vroughness = param.GetFloat("vroughness", 0.f);
    return std::make_shared<Material>(Material::GLOSSY_TRANSMISSION | Material::GLOSSY_REFLECT, 
        Kr, Kt, eta, uroughness, vroughness);
}


