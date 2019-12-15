#pragma once
#ifndef __CUDASCENE_H

#include "renderer/core/scene.h"

class CUDAScene {    
public:
    __device__ __host__
    CUDAScene();
    __device__ __host__
    CUDAScene(Scene* scene);

    __device__ __host__
    bool Intersect(const Ray& ray, Interaction* interaction) const;

    __device__ __host__
    Spectrum Shading(const Interaction& interaction) const;


    TriangleMesh* m_triangleMeshes;
    int m_triangleMeshNum;
    Triangle* m_triangles;
    int m_triangleNum;
    Material* m_materials;
    int m_materialNum;
    Primitive* m_primitives;
    int m_primitiveNum;
    Light* m_lights;
    int m_lightNum;
};

inline __device__ __host__
CUDAScene::CUDAScene()
{
    m_triangleMeshes = nullptr;
    m_triangleMeshNum = 0;
    m_triangles = nullptr;
    m_triangleNum = 0;
    m_materials = nullptr;
    m_materialNum = 0;
    m_primitives = nullptr;
    m_primitiveNum = 0;
    m_lights = nullptr;
    m_lightNum = 0;
}

inline __device__ __host__
CUDAScene::CUDAScene(Scene* scene) {
    // Move TriangleMesh Data
    m_triangleMeshNum = scene->m_triangleMeshes.size();
    m_triangleMeshes = new TriangleMesh[m_triangleMeshNum];
    for (int i = 0; i < m_triangleMeshNum; i++) {
        memcpy(&m_triangleMeshes[i], &scene->m_triangleMeshes[i], sizeof(TriangleMesh));
    }

    // Move Triangle Data
    m_triangleNum = scene->m_triangles.size();

    // Move Material Data
    m_materialNum = scene->m_materials.size();

    // Move Light Data
    m_lightNum = scene->m_lights.size();

    // Move Primitive Data
    m_primitiveNum = scene->m_primitives.size();
}

inline __device__ __host__
bool CUDAScene::Intersect(const Ray& ray, Interaction* interaction) const
{
    Float tHit;
    bool ret_hit = false;
    for (int i = 0; i < m_primitiveNum; i++) {
        int triangleID = m_primitives[i].m_shapeID;
        bool hit = m_triangles[triangleID].Intersect(ray, &tHit, interaction);
        if (hit) {
            ret_hit = true;
            ray.tMax = tHit;
            interaction->m_primitiveID = i;
        }
    }
    return ret_hit;
}

inline __device__ __host__ 
Spectrum CUDAScene::Shading(const Interaction& interaction) const
{
    const int& primID = interaction.m_primitiveID;
    const int& mtlID = m_primitives[primID].m_materialID;
    return m_materials[mtlID].m_Kd;
}



#endif // __CUDASCENE_H