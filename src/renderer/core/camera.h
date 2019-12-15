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

    Ray GenerateRay(const Point2f& p) const;

    Float m_fov;
    Transform m_cameraToWorld, m_worldToCamera;
    Transform m_rasterToCamera;

    Film m_film;
};

std::shared_ptr<Camera>
CreateCamera(
    const ParameterSet& param,
    const Film& film, 
    const Transform objToWorld,
    const Transform worldToObj);

#endif // !__CAMERA_H
