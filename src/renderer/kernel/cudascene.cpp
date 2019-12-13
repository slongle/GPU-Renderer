#include "cudascene.h"

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
    m_lightNum= scene->m_lights.size();

    // Move Primitive Data
    m_primitiveNum = scene->m_primitives.size();
}
