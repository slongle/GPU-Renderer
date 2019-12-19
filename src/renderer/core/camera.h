#pragma once
#ifndef __CAMERA_H
#define __CAMERA_H

#include "renderer/core/fwd.h"
#include "renderer/core/transform.h"
#include "renderer/core/parameterset.h"
#include "renderer/core/film.h"

class Camera {
public:
    Camera() {}
    Camera(
        Float fov,
        Film film,
        Transform objToWorld,
        Transform worldToObj);

    __host__ __device__ 
    Ray GenerateRay(const Point2f& p) const;

    Float m_fov;
    Transform m_cameraToWorld, m_worldToCamera;
    Transform m_rasterToCamera, m_cameraToRaster;

    Film m_film;
};

std::shared_ptr<Camera>
CreateCamera(
    const ParameterSet& param,
    const Film& film,
    const Transform objToWorld,
    const Transform worldToObj);


inline __device__ __host__
Ray Camera::GenerateRay(
    const Point2f& p) const
{
    Point3f pCamera = m_rasterToCamera(Point3f(p.x, p.y, 0));
    Ray r(Point3f(), Normalize(Vector3f(pCamera)));
    Ray ray = m_cameraToWorld(r);
    return ray;
}

#endif // !__CAMERA_H
