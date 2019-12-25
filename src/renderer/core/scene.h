#pragma once
#ifndef __SCENE_H
#define __SCENE_H

#include "renderer/core/fwd.h"
#include "renderer/core/geometry.h"
#include "renderer/core/primitive.h"
#include "renderer/core/interaction.h"
#include "renderer/core/bvh.h"
#include <vector>



class Scene {
public:
    Scene() {}

    bool Intersect(const Ray& ray) const;
    bool IntersectP(const Ray& ray, Interaction* interaction) const;

    int AddTriangleMesh(TriangleMesh triangleMesh);
    std::pair<int, int> AddTriangles(std::vector<std::shared_ptr<Triangle>> triangles);
    int AddMaterial(std::shared_ptr<Material> material);
    int AddLight(std::shared_ptr<Light> light);
    void AddPrimitive(Primitive p);
    void preprocess();
    std::vector<TriangleMesh> m_triangleMeshes;
    std::vector<Triangle> m_triangles;
    std::vector<Material> m_materials;
    std::vector<Light> m_lights;
    std::vector<Primitive> m_primitives;
    BVH* m_shapeBvh;
};


inline
bool Scene::Intersect(const Ray& ray) const
{
    return m_shapeBvh->Intersect(ray);
}
inline
bool Scene::IntersectP(const Ray& ray, Interaction* interaction) const
{
    return m_shapeBvh->IntersectP(ray, interaction);
}



/*
inline
bool Scene::IntersectP(const Ray& ray, Interaction* interaction) const
{
    Float tHit;
    bool ret_hit = false;
    for (int i = 0; i < m_primitives.size(); i++) {
        int triangleID = m_primitives[i].m_shapeID;
        bool hit = m_triangles[triangleID].IntersectP(ray, &tHit, interaction);
        if (hit) {
            ret_hit = true;
            ray.tMax = tHit;
            interaction->m_primitiveID = i;
//            printf("   %d\n", i);
        }
    }
    return ret_hit;
}
*/




#endif // !__SCENE_H
