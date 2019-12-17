#pragma once
#ifndef __CUDARENDERER_H
#define __CUDARENDERER_H

#include "renderer/core/renderer.h"
#include "renderer/kernel/cudascene.h"

class CUDARenderer {
public:
    CUDARenderer(
        Integrator* integrator,
        Camera* camera,
        CUDAScene* scene) 
        : m_camera(camera), m_integrator(integrator), m_scene(scene) {}

    Camera* m_camera;
    Integrator* m_integrator;
    CUDAScene* m_scene;
};

#endif // !__CUDARENDERER_H
