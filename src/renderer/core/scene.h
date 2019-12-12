#pragma once
#ifndef __SCENE_H
#define __SCENE_H

#include "renderer/core/fwd.h"
#include "renderer/core/geometry.h"
#include "renderer/core/primitive.h"

#include <vector>


class Scene {
public:
    Scene() {}


    int AddTriangleMesh(TriangleMesh triangleMesh);
    std::pair<int, int> AddTriangles(std::vector<std::shared_ptr<Triangle>> triangles);
    int AddMaterial(std::shared_ptr<Material> material);
    int AddLight(std::shared_ptr<Light> light);
    void AddPrimitive(Primitive p);

    std::vector<TriangleMesh> m_triangleMeshes;
    std::vector<Triangle> m_triangles;
    std::vector<Material> m_materials;
    std::vector<Light> m_lights;
    std::vector<Primitive> m_primitives;
};

#endif // !__SCENE_H
