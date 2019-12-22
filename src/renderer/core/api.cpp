#include "api.h"

#include "renderer/core/transform.h"
#include "renderer/core/primitive.h"
#include "renderer/core/scene.h"
#include "renderer/core/integrator.h"
#include "renderer/core/renderer.h"
#include "renderer/core/camera.h"
#include "renderer/core/film.h"
#include "renderer/core/triangle.h"
#include "renderer/core/material.h"


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

    int GetNamedMaterial(
        const std::string& name) const;

    int MakeMaterial(
        const std::string& type,
        const ParameterSet& params);

    std::pair<int, int> MakeShape(
        const std::string& type,
        const ParameterSet& params);

    int MakeLight(
        const std::string& type,
        const ParameterSet& params,
        int triangleID);

    void MakeCamera();
    void MakeFilm();
    void MakeIntegrator();
    void MakeRenderer();

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
    Transform m_cameraTransform;

    Camera m_camera;
    Film m_film;
    Integrator m_integrator;
    std::shared_ptr<Renderer> m_renderer;

    bool m_hasAreaLight;
    std::string m_areaLightType;
    ParameterSet m_areaLightParameterSet;

    int m_currentMaterial;
    std::map<std::string, int> m_namedMaterials;
    int m_currentMedium;
    std::map<std::string, int> m_namedMedium;

    Scene m_scene;

};

static std::unique_ptr<Options> options(new Options);

void apiAttributeBegin()
{
    options->m_transformStack.push_back(options->m_currentTransform);
    options->m_currentTransform.Identity();
}

void apiAttributeEnd()
{
    options->m_currentTransform = options->m_transformStack.back();
    options->m_transformStack.pop_back();
}

void apiWorldBegin() 
{
    options->m_transformStack.push_back(options->m_currentTransform);
    options->m_currentTransform.Identity();
}

std::shared_ptr<Renderer> 
apiWorldEnd()
{
    /*
     * Camera
     *   -- Film
     *        |- Filter
     *        -- Sampler
     */

    options->MakeFilm();
    options->MakeCamera();
    options->MakeIntegrator();    
    options->MakeRenderer();
    return options->m_renderer;
}

void apiTransformBegin()
{
    options->m_transformStack.push_back(options->m_currentTransform);
    options->m_currentTransform.Identity();
}

void apiTransformEnd()
{
    options->m_currentTransform = options->m_transformStack.back();
    options->m_transformStack.pop_back();
}

void apiTransform(const Float m[16])
{
    //Transform t(m);
    Transform t(Matrix4x4(
        m[0], m[4], m[8], m[12], m[1], m[5], m[9], m[13], m[2],
        m[6], m[10], m[14], m[3], m[7], m[11], m[15]));
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
    options->m_cameraTransform = options->m_currentTransform;
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
    std::pair<int,int> shapes = options->MakeShape(type, params);
    int mtlID = options->m_currentMaterial;
    for (int shapeID = shapes.first; shapeID < shapes.second; shapeID++) {        
        int areaLightID = -1;
        if (options->m_hasAreaLight) {
            areaLightID = options->MakeLight(options->m_areaLightType, 
                options->m_areaLightParameterSet, shapeID);
        }
        options->m_scene.AddPrimitive(Primitive(shapeID, mtlID, areaLightID));
    }
    if (options->m_hasAreaLight) {
        options->m_hasAreaLight = false;
    }
}

void apiAreaLightSource(const std::string& type, ParameterSet params)
{
    options->m_hasAreaLight = true;
    options->m_areaLightType = type;
    options->m_areaLightParameterSet = params;
}

void 
Options::MakeNamedMaterial(
    const std::string& name, 
    const ParameterSet& params)
{
    std::string type = params.GetString("type");
    int mtl = MakeMaterial(type, params);
    m_namedMaterials[name] = mtl;
}

int 
Options::GetNamedMaterial(
    const std::string& name) const
{
    ASSERT(m_namedMaterials.count(name), "No named material " + name);
    return m_namedMaterials.find(name)->second;
}

int Options::MakeMaterial(
    const std::string& type, 
    const ParameterSet& params)
{
    std::shared_ptr<Material> mtl;
    if (type == "diffuse" || type == "matte") {
        mtl = CreateMatteMaterial(params);
    }
    else if (type == "metal") {
        mtl = CreateMetalMaterial(params);
    }
    else if (type == "glass") {
        mtl = CreateGlassMaterial(params);
    }
    else {
        ASSERT(0, "Can't support material " + type);
    }
    int mtlID = m_scene.AddMaterial(mtl);
    return mtlID;
}

std::pair<int, int> Options::MakeShape(
    const std::string& type,
    const ParameterSet& params)
{
    Transform objToWorld = m_currentTransform;
    Transform worldToObj = Inverse(objToWorld);
    std::vector<std::shared_ptr<Triangle>> triangles;
    if (type == "trianglemesh") {
        triangles = CreateTriangleMeshShape(params, objToWorld, worldToObj);
    }
    else if (type == "plymesh") {
        triangles = CreatePLYMeshShape(params, objToWorld, worldToObj);
    }
    else if (type == "sphere") {
        triangles = CreateSphereShape(params, objToWorld, worldToObj);
    }
    else {
        ASSERT(0, "Can't support shape " + type);
    }
    std::pair<int, int> interval = m_scene.AddTriangles(triangles);
    return interval;
}

int Options::MakeLight(
    const std::string& type, 
    const ParameterSet& params, 
    int shapeID)
{
    std::shared_ptr<Light> light;
    if (type == "area" || type == "diffuse"){
        light = CreateAreaLight(params, shapeID);
    }
    int lightID = m_scene.AddLight(light);
    return lightID;
}

void Options::MakeCamera()
{
    Transform worldToObj = m_cameraTransform; 
    Transform objToWorld = Inverse(worldToObj);
    m_camera = *CreateCamera(m_cameraParameterSet, m_film, objToWorld, worldToObj);
}

void Options::MakeFilm() {
    m_film = *CreateFilm(m_filmParameterSet);
}

void Options::MakeIntegrator()
{
    m_integrator = *CreateIntegrator(m_integratorParameterSet);    
    int nSample = m_samplerParameterSet.GetInt("pixelsamples");
    m_integrator.m_nSample = nSample;
}

void Options::MakeRenderer()
{    
    m_renderer = std::make_shared<Renderer>();
    m_renderer->m_scene = m_scene;
    m_renderer->m_camera = m_camera;
    m_renderer->m_integrator = m_integrator;    
}









