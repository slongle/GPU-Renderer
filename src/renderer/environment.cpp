#include "environment.h"

EnvironmentLightView::EnvironmentLightView(const EnvironmentLight* light) 
    : m_has(light->m_has), m_width(light->m_width), m_height(light->m_height), 
    m_buffer(light->m_gpu_buffer.data()),
    m_o2w(light->m_o2w), m_w2o(light->m_w2o), 
    m_world_center(light->m_world_center)
{
}

EnvironmentLight::EnvironmentLight(
    const std::string& filename, 
    Transform o2w)
    : m_has(true), m_o2w(o2w), m_w2o(Inverse(o2w))
{
    Buffer<HOST_BUFFER, float> buffer;
    ReadImage(filename, &m_width, &m_height, buffer);
    m_cpu_buffer.resize(m_width * m_height);
    for (int i = 0; i < m_width * m_height * 3; i += 3) {
        m_cpu_buffer[i / 3] = make_float3(buffer[i + 0], buffer[i + 1], buffer[i + 2]);
    }
    m_gpu_buffer.copyFrom(m_cpu_buffer.size(), HOST_BUFFER, m_cpu_buffer.data());
}

void EnvironmentLight::setup(
    const std::string& filename,
    Transform o2w)
{    
    m_has = true;
    m_o2w = o2w, m_w2o = Inverse(o2w);
    Buffer<HOST_BUFFER, float> buffer;
    ReadImage(filename, &m_width, &m_height, buffer);
    m_cpu_buffer.resize(m_width * m_height);
    for (int i = 0; i < m_width * m_height * 3; i += 3) {
        m_cpu_buffer[i / 3] = make_float3(buffer[i + 0], buffer[i + 1], buffer[i + 2]);
    }
    m_gpu_buffer.copyFrom(m_cpu_buffer.size(), HOST_BUFFER, m_cpu_buffer.data());        
}
