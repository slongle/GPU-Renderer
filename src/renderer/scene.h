#pragma once

#include <string>
#include <vector>

#include "renderer/triangle.h"
#include "renderer/camera.h"
#include "renderer/ray.h"
#include "renderer/bvh.h"

class Scene;

struct SceneView 
{
    SceneView(const Scene* scene);

    Camera m_camera;
    Triangle* m_triangles;
    uint32 m_triangles_num;
    Triangle* m_lights;
    uint32 m_lights_num;
    BVHLinearNode* m_bvh;
};

class Scene {
public:
    Scene() {}
    Scene(const std::string& filename);
    ~Scene() { }

    void createDeviceData();
    SceneView view() const { return SceneView(this); }
    
public:
    Camera m_camera;

    std::vector<Triangle> m_cpu_triangles;
    std::vector<Triangle> m_cpu_lights;
    Buffer<DEVICE_BUFFER, Triangle> m_gpu_triangles;
    Buffer<DEVICE_BUFFER, Triangle> m_gpu_lights;

    std::vector<BVHLinearNode> m_cpu_bvh;
    Buffer<DEVICE_BUFFER, BVHLinearNode> m_gpu_bvh;
};