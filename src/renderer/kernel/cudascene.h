#pragma once
#ifndef __CUDASCENE_H

#include "renderer/core/scene.h"

class CUDAScene {    
public:
    CUDAScene(Scene* scene);

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

#endif // __CUDASCENE_H