#include "material.h"

Material::Material(
    ComponentType type,
    Spectrum Kd)
    : m_type(type), m_diffuseReflect(Kd)
{
}

Material::Material(
    ComponentType type,
    const Spectrum& eta,
    const Spectrum& k,
    const Float& uroughness,
    const Float& vroughness)
    : m_type(type), m_glossyReflect(Spectrum(1), true, Spectrum(1), eta, k, uroughness, vroughness)
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


