#pragma once

#include "renderer/fwd.h"
#include "renderer/bsdf.h"
#include "renderer/scene.h"
#include "renderer/ray.h"

class Triangle;

class Vertex 
{
public:
    HOST_DEVICE
    Vertex() {}

    HOST_DEVICE
        void setup(
            Ray ray,
            Hit hit,
            SceneView scene)
    {     
        const int tri_ID = hit.triangle_id;
        Triangle& triangle = scene.m_triangles[tri_ID];

        triangle.setupDifferential(make_float2(hit.u, hit.v), &m_geom);                
        m_wo = -normalize(ray.d);

        Material& material = triangle.m_mesh.m_material;
        m_bsdf = BSDF(m_geom, material);
    }

    Differential m_geom;

    float3 m_wo;
    float3 m_wi;
    BSDF m_bsdf;
};