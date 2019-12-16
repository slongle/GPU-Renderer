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


            Ray ray = camera.GenerateRay(Point2f(x + NextRandom(seed), y + NextRandom(seed)));
            Interaction inter;


            bool hit = scene.Intersect(ray, &inter);
            if (std::fabs(inter.m_geometryN.x) <EPSILON && 
                std::fabs(inter.m_geometryN.y) <EPSILON && 
                std::fabs(inter.m_geometryN.z) <EPSILON) {
                int a = 1;
                a++;
            }
            Spectrum L;
            if (hit) {
                printf("%d %d %d\n", inter.m_geometryN.x, inter.m_geometryN.y, inter.m_geometryN.z);
            }

            camera.m_film.SetVal(x, y, L);
        }
    }
    camera.m_film.Output();
}

#endif // !__CPURENDERER_H
