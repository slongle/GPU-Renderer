#pragma once
#ifndef __RENDERER_H
#define __RENDERER_H

#include "renderer/core/scene.h"
#include "renderer/core/camera.h"
#include "renderer/core/integrator.h"

class Renderer {
public:
    Renderer() {}

    Scene m_scene;
    Camera m_camera;
    Integrator m_integrator;
};

#endif // !__RENDERER_H
