#include "camera.h"

Camera::Camera(
    Float fov,
    Film film,
    Transform objToWorld,
    Transform worldToObj)
    :m_fov(fov), m_film(film), m_cameraToWorld(objToWorld), m_worldToCamera(worldToObj)
{
    Point2i resolution = film.m_resolution;
    Float aspect = static_cast<Float>(resolution.x) / resolution.y;
    Point2f frame = aspect > 1 ? Point2f(aspect, 1) : Point2f(1, 1 / aspect);
    Transform cameraToScreen = Perspective(m_fov, 1e-2f, 1000.f);
    Transform screenToRaster =
        Scale(resolution.x, resolution.y, 1) *
        Scale(1 / (2 * frame.x), -1 / (2 * frame.y), 1) *
        Translate(frame.x, -frame.y, 0);
    m_rasterToCamera = Inverse(screenToRaster * cameraToScreen);
}

std::shared_ptr<Camera>
CreateCamera(
    const ParameterSet& param,
    const Film& film,
    const Transform objToWorld,
    const Transform worldToObj)
{
    Float fov = param.GetFloat("fov");
    //Point2i resolution = film.m_resolution;
    //Float aspect = static_cast<Float>(resolution.x) / resolution.y;
    return std::make_shared<Camera>(fov, film, objToWorld, worldToObj);
}
