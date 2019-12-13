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
        Transform objToWorld,
        Transform worldToObj);


    Float m_fov;
    Transform m_objToWorld, m_worldToObj;

    Film m_film;
};

std::shared_ptr<Camera>
CreateCamera(
    const ParameterSet& param,
    const Transform objToWorld,
    const Transform worldToObj);

#endif // !__CAMERA_H
