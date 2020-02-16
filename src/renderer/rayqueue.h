#pragma once

#include "utility/CUDA/warp_atomic.h"
#include "renderer/ray.h"

class RayQueue {
public:
    RayQueue() {}

    Ray* m_rays;
    Hit* m_hits;
    Spectrum* m_weight;
    float* m_pdf;
    bool* m_specular;
    uint32* m_idx;
    uint32* m_seed;
    uint32* m_size;
    
    __device__
    void append(
        const Ray& ray,
        const Spectrum& weight,
        const float& pdf,
        const bool& specular,
        const uint32& pixel_idx)
    {
        const uint32 slot = warp_increment(m_size);
        m_rays[slot] = ray;
        m_weight[slot] = weight;
        m_pdf[slot] = pdf;
        m_specular[slot] = specular;
        m_idx[slot] = pixel_idx;
    }    

    __device__
    void append(
        const Ray&      ray, 
        const Spectrum& weight, 
        const bool&     specular,
        const uint32&   pixel_idx) 
    {
        const uint32 slot = warp_increment(m_size);
        m_rays[slot] = ray;
        m_weight[slot] = weight;
        m_specular[slot] = specular;
        m_idx[slot] = pixel_idx;
    }
};