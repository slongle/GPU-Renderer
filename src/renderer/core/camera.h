#pragma once
#ifndef __CAMERA_H
#define __CAMERA_H

#include "renderer/core/fwd.h"
#include "renderer/core/transform.h"
#include "renderer/core/parameterset.h"
#include "renderer/core/film.h"

class Camera {
public:
    __host__ __device__ Camera() {}
    __host__ __device__ Camera(
        Float fov,
        Film film,
        Transform objToWorld,
        Transform worldToObj);

    __host__ __device__ 
    Ray GenerateRay(const Point2f& p) const;

    Float m_fov;
    Transform m_cameraToWorld, m_worldToCamera;
    Transform m_rasterToCamera;

    Film m_film;
};

inline
std::shared_ptr<Camera>
CreateCamera(
    const ParameterSet& param,
    const Film& film,
    const Transform objToWorld,
    const Transform worldToObj);

inline __device__ __host__
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

inline __device__ __host__
Ray Camera::GenerateRay(
    const Point2f& p) const
{
    Point3f pCamera = m_rasterToCamera(Point3f(p.x, p.y, 0));
    Ray r(Point3f(), Normalize(Vector3f(pCamera)));
    Ray ray = m_cameraToWorld(r);
    return ray;
}

inline
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

#endif // !__CAMERA_H
