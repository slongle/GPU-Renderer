#include "scene.h"
#include "renderer/loaders/loader.h"

Scene::Scene(const std::string& filename)
{
    load_scene(filename, this);
    for (uint32 i = 0; i < m_meshes.size(); i++) 
    {
        TriangleMeshView mesh_view = m_meshes[i].view(true);
        for (uint32 j = 0; j < mesh_view.m_triangle_num; j++) 
        {
            m_cpu_triangles.emplace_back(j * 3, mesh_view);
        }
    }
    for (int i = 0; i < m_cpu_triangles.size(); i++) {
        if (m_cpu_triangles[i].m_mesh.m_material.isEmission()) {
            m_cpu_lights.push_back(m_cpu_triangles[i]);
        }
    }    
    std::cout << "# of light : " << m_cpu_lights.size() << std::endl;
    m_cpu_bvh = LBVH_build(m_cpu_triangles);
    
    AABB world_box = m_cpu_bvh[0].m_box;
    m_environment_light.m_world_center = world_box.centroid();
}

void Scene::createDeviceData()
{
    uint32 offset = 0;
    for (uint32 i = 0; i < m_meshes.size(); i++)
    {
        m_meshes[i].createDeviceData();
        TriangleMeshView mesh_view = m_meshes[i].view();
        for (uint32 j = 0; j < mesh_view.m_triangle_num; j++)
        {
            m_cpu_triangles[offset++].m_mesh = mesh_view;
        }
    }
    m_cpu_lights.clear();
    for (int i = 0; i < m_cpu_triangles.size(); i++) {
        if (m_cpu_triangles[i].m_mesh.m_material.isEmission()) {
            m_cpu_lights.push_back(m_cpu_triangles[i]);
        }
    }
    m_gpu_triangles.copyFrom(m_cpu_triangles.size(), HOST_BUFFER, m_cpu_triangles.data());
    m_gpu_lights.copyFrom(m_cpu_lights.size(), HOST_BUFFER, m_cpu_lights.data());
    m_gpu_bvh.copyFrom(m_cpu_bvh.size(), HOST_BUFFER, m_cpu_bvh.data());
}

SceneView::SceneView(const Scene* scene) :
    m_camera(scene->m_camera),
    m_triangles(scene->m_gpu_triangles.data()),
    m_triangles_num(scene->m_gpu_triangles.size()),
    m_lights(scene->m_gpu_lights.data()),
    m_lights_num(scene->m_gpu_lights.size()),
    m_environment_light(scene->m_environment_light.view()),
    m_bvh(scene->m_gpu_bvh.data())
{}
