#pragma once
#ifndef __RENDERER_H
#define __RENDERER_H

#include "renderer/core/fwd.h"
#include "renderer/core/scene.h"
#include "renderer/core/camera.h"
#include "renderer/core/integrator.h"

class Renderer {
public:
    Renderer() {}


    std::shared_ptr<Scene> m_scene;
    std::shared_ptr<Integrator> m_integrator;
    std::shared_ptr<Camera> m_camera;
};

#endif // !__RENDERER_H
