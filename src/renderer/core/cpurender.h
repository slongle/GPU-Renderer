#pragma once
#ifndef __CPURENDERER_H
#define __CPURENDERER_H

#include "renderer/core/renderer.h"
#include "renderer/core/film.h"
#include "renderer/core/sampling.h"

inline void render(std::shared_ptr<Renderer> renderer)
{
    Camera& camera = renderer->m_camera;
    Scene& scene = renderer->m_scene;
    for (int x = 0; x < camera.m_film.m_resolution.x; x++) {
        for (int y = 0; y < camera.m_film.m_resolution.y; y++) {
            int index = y * camera.m_film.m_resolution.x + x;
            unsigned int seed = RandomInit(index, 0);

            //Point2f p(512.03, 512.04);
            Ray ray = camera.GenerateRay(Point2f(x + NextRandom(seed), y + NextRandom(seed)));    
            //Ray ray = camera.GenerateRay(p);
            //cout<<ray
            Interaction inter;
            bool hit = scene.Intersect(ray, &inter);
            Spectrum L;
            if (hit) {
                L += scene.Shading(inter);
            }

            camera.m_film.SetVal(x, y, L);
        }
    }
    camera.m_film.Output();
}

#endif // !__CPURENDERER_H
