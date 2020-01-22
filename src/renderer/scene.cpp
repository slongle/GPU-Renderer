#include "scene.h"
#include "renderer/loaders/loader.h"

Scene::Scene(const std::string& filename)
{
    load_scene(filename, m_cpu_triangles);
    for (int i = 0; i < m_cpu_triangles.size(); i++) {
        if (m_cpu_triangles[i].m_material.isEmission()) {
            m_cpu_lights.push_back(m_cpu_triangles[i]);
        }
    }
    m_cpu_bvh = LBVH_build(m_cpu_triangles);
}

void Scene::createDeviceData()
{
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
    m_bvh(scene->m_gpu_bvh.data())
{}
