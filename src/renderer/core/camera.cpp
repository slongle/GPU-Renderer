#include "camera.h"

Camera::Camera(
    Float fov,
    Transform objToWorld,
    Transform worldToObj)
    :m_fov(fov), m_objToWorld(objToWorld), m_worldToObj(worldToObj)
{
}

std::shared_ptr<Camera> 
CreateCamera(
    const ParameterSet& param, 
    const Transform objToWorld, 
    const Transform worldToObj)
{
    Float fov = param.GetFloat("fov");
    return std::make_shared<Camera>(fov, objToWorld, worldToObj);
}

