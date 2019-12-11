#pragma once
#ifndef __SCENE_H
#define __SCENE_H

#include "renderer/core/fwd.h"
#include "renderer/core/geometry.h"

#include <vector>


class Scene {
public:
    Scene() {}

    std::vector<std::shared_ptr<Light>> m_lights;
    std::vector<std::shared_ptr<Primitive>> m_primitives;
};

#endif // !__SCENE_H
