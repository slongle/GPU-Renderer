#pragma once

#include "renderer/fwd.h"
#include "renderer/bsdf.h"
#include "renderer/scene.h"

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
        m_p = ray.o + ray.d * hit.t;

        const int tri_ID = hit.triangle_id;
        Triangle* triangle = &scene.m_triangles[tri_ID];
        float3 p0, p1, p2;
        triangle->getVertices(p0, p1, p2);

        const float3 dp_du = p0 - p2;
        const float3 dp_dv = p1 - p2;
        m_normal_g = normalize(cross(dp_du, dp_dv));
        if (triangle->m_mesh.m_index[triangle->m_index].z != -1)
        {
            float3 n0, n1, n2;
            triangle->getNormals(n0, n1, n2);
            m_normal_s = normalize(n0 * (1 - hit.u - hit.v) + n1 * hit.u + n2 * hit.v);
        }
        else
        {
            m_normal_s = m_normal_g;
        }
        //m_normal_s = m_normal_g;
        
        m_wo = -normalize(ray.d);

        const Material& material = triangle->m_mesh.m_material;
        m_bsdf = BSDF(m_normal_g, m_normal_s, material);
    }

    float3 m_p;
    float3 m_normal_g;
    float3 m_normal_s;
    float3 m_wo;
    float3 m_wi;
    BSDF m_bsdf;
};