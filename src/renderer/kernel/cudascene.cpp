#include "cudascene.h"

CUDAScene::CUDAScene(Scene* scene) {
    // Move TriangleMesh Data
    m_triangleMeshNum = scene->m_triangleMeshes.size();

    // Move Triangle Data
    m_triangleNum = scene->m_triangles.size();

    // Move Material Data
    m_materialNum = scene->m_materials.size();
    
    // Move Light Data
    m_lightNum= scene->m_lights.size();

    // Move Primitive Data
    m_primitiveNum = scene->m_primitives.size();
}
