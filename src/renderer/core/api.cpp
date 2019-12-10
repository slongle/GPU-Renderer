#include "api.h"

#include "renderer/core/transform.h"

#include "renderer/core/shape.h"

#include "renderer/light/arealight.h"


class Options {
public:
    Options() {
        m_currentTransform.Identity();
        m_hasAreaLight = false;
    }

    void 
        MakeNamedMaterial(
        const std::string& name, 
        const ParameterSet& params);

    std::shared_ptr<Material> 
        GetNamedMaterial(
        const std::string& name);

    std::vector<std::shared_ptr<Shape>>
        MakeShape(const std::string& type,
            const ParameterSet& params,
            const Transform& objToWorld);


    Transform m_currentTransform;
    std::vector<Transform> m_transformStack;

    std::string m_integratorType;
    ParameterSet m_integratorParameterSet;
    std::string m_samplerType;
    ParameterSet m_samplerParameterSet;
    std::string m_filterType;
    ParameterSet m_filterParameterSet;
    std::string m_filmType;
    ParameterSet m_filmParameterSet;
    std::string m_cameraType;
    ParameterSet m_cameraParameterSet;

    bool m_hasAreaLight;
    std::string m_areaLightType;
    ParameterSet m_areaLightParameterSet;

    std::shared_ptr<Material> m_currentMaterial;
    std::map<std::string, std::shared_ptr<Material>> m_namedMaterials;
    std::shared_ptr<Medium> m_currentMedium;
    std::map<std::string, std::shared_ptr<Medium>> m_namedMedium;
};

static std::unique_ptr<Options> options(new Options);

void apiAttributeBegin()
{
    options->m_transformStack.push_back(options->m_currentTransform);
}

void apiAttributeEnd()
{
    options->m_currentTransform = options->m_transformStack.back();
    options->m_transformStack.pop_back();
}

void apiTransform(const Float m[16])
{
    Transform t(m);
    options->m_currentTransform *= t;
}

void apiIntegrator(const std::string& type, ParameterSet params)
{
    options->m_integratorType = type;
    options->m_integratorParameterSet = params;
}

void apiSampler(const std::string& type, ParameterSet params)
{
    options->m_samplerType = type;
    options->m_samplerParameterSet = params;
}

void apiFilter(const std::string& type, ParameterSet params)
{
    options->m_filterType = type;
    options->m_filterParameterSet = params;
}

void apiFilm(const std::string& type, ParameterSet params) {
    options->m_filmType = type;
    options->m_filmParameterSet = params;
}

void apiCamera(const std::string& type, ParameterSet params)
{
    options->m_cameraType = type;
    options->m_cameraParameterSet = params;
}

void apiNamedMaterial(const std::string& name, ParameterSet params)
{
    options->m_currentMaterial = options->GetNamedMaterial(name);
}

void apiMakeNamedMaterial(const std::string& name, ParameterSet params)
{
    options->MakeNamedMaterial(name, params);
}

void apiShape(const std::string& type, ParameterSet params)
{
    std::vector<std::shared_ptr<Primitive>> primitives;
    std::vector<std::shared_ptr<AreaLight>> areaLights;
    std::vector<std::shared_ptr<Shape>> shapes = options->MakeShape(
        type, params, options->m_currentTransform);
    std::shared_ptr<Material> mtl = options->m_currentMaterial;
    for (auto s : shapes) {

    }
}

void apiAreaLightSource(const std::string& type, ParameterSet params)
{
    options->m_areaLightType = type;
    options->m_areaLightParameterSet = params;
}

void 
Options::MakeNamedMaterial(
    const std::string& name, 
    const ParameterSet& params)
{
}

std::shared_ptr<Material> 
Options::GetNamedMaterial(
    const std::string& name)
{
    return std::shared_ptr<Material>();
}

std::vector<std::shared_ptr<Shape>> 
Options::MakeShape(
    const std::string& type, 
    const ParameterSet& params, 
    const Transform& objToWorld)
{
    return std::vector<std::shared_ptr<Shape>>();
}
