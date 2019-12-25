#include "scene.h"

void Scene::Preprocess()
{
    m_shapeBvh->Build(m_primitives, m_triangles);
}

int Scene::AddTriangleMesh(TriangleMesh triangleMesh)
{
    int ID = m_triangleMeshes.size();
    m_triangleMeshes.push_back(triangleMesh);
    return ID;
}

std::pair<int, int>
Scene::AddTriangles(std::vector<std::shared_ptr<Triangle>> triangles)
{
    int meshID = AddTriangleMesh(*triangles[0]->m_triangleMeshPtr);
    std::pair<int, int> interval(m_triangles.size(), m_triangles.size() + triangles.size());
    for (int i = 0; i < triangles.size(); i++) {
        triangles[i]->m_triangleMeshID = meshID;
        m_triangles.push_back(*triangles[i]);
    }
    return interval;
}

int Scene::AddMaterial(std::shared_ptr<Material> material)
{
    int ID = m_materials.size();
    m_materials.push_back(*material);
    return ID;
}


int Scene::AddLight(std::shared_ptr<Light> light)
{
    int ID = m_lights.size();
    m_lights.push_back(*light);
    return ID;
}

void Scene::AddPrimitive(Primitive p)
{
    m_primitives.push_back(p);
}