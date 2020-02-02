#pragma once

#include "renderer/fwd.h"
#include "renderer/spectrum.h"
#include "renderer/ray.h"
#include "renderer/transform.h"

class EnvironmentLight;

class EnvironmentLightView
{
public:
    EnvironmentLightView(const EnvironmentLight* light);

    HOST_DEVICE
    Spectrum Le(const Ray& ray) const
    {
        float3 d = normalize(ray.o + 100000 * ray.d - m_world_center);
        d = normalize(m_w2o.transformVector(d));
        float theta = spherical_theta(d);
        float phi = spherical_phi(d);
        float u = phi* INV_2_PI, v = theta * INV_PI;
        int u2 = u * m_width, v2 = v * m_height;
        int idx = v2 * m_width + u2;
        float3 col = m_buffer[idx];
        return col;
    }

public:
    bool m_has;
    float3* m_buffer;
    int m_width, m_height;
    Transform m_o2w, m_w2o;
    float3 m_world_center;
};

class EnvironmentLight
{
public:
    EnvironmentLight() :m_has(false) {}
    EnvironmentLight(const std::string& filename, Transform o2w);
    void setup(const std::string& filename, Transform o2w);
    EnvironmentLightView view() const { return EnvironmentLightView(this); }

public:
    bool m_has;
    Buffer<HOST_BUFFER, float3> m_cpu_buffer;
    Buffer<DEVICE_BUFFER, float3> m_gpu_buffer;
    int m_width, m_height;
    Transform m_o2w, m_w2o;
    float3 m_world_center;
};