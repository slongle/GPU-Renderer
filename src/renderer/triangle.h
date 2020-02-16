#pragma once

#include "renderer/fwd.h"
#include "renderer/material.h"
#include "renderer/sampling.h"
#include "renderer/aabb.h"

class TriangleMesh;

class TriangleMeshView
{
public:
    uint32 m_triangle_num;
    uint32 m_p_num, m_n_num, m_uv_num, m_index_num;
    const float3* m_p;
    const float3* m_n;
    const float2* m_uv;
    const int3* m_index;

    Material m_material;

    TriangleMeshView() {}
    TriangleMeshView(const TriangleMesh* mesh);
    TriangleMeshView(const TriangleMesh* mesh, bool host);
};

class TriangleMesh
{
public:
    TriangleMesh() {}

    uint32 m_triangle_num;
    std::vector<float3> m_cpu_p;
    std::vector<float3> m_cpu_n;
    std::vector<float2> m_cpu_uv;
    std::vector<int3>   m_cpu_index;

    Buffer<DEVICE_BUFFER, float3> m_gpu_p;
    Buffer<DEVICE_BUFFER, float3> m_gpu_n;
    Buffer<DEVICE_BUFFER, float2> m_gpu_uv;
    Buffer<DEVICE_BUFFER, int3>   m_gpu_index;

    Material m_material;

    void createDeviceData();
    TriangleMeshView view() const { return TriangleMeshView(this); }
    TriangleMeshView view(bool host) const { return TriangleMeshView(this, host); }
};

class Triangle {
public:
    Triangle() {}

    Triangle(
        uint32 index,
        TriangleMeshView mesh);

    HOST_DEVICE
    void getNormals(float3& n0, float3& n1, float3& n2) const
    {
        n0 = m_mesh.m_n[m_mesh.m_index[m_index + 0].z];
        n1 = m_mesh.m_n[m_mesh.m_index[m_index + 1].z];
        n2 = m_mesh.m_n[m_mesh.m_index[m_index + 2].z];
    }

    HOST_DEVICE
    void getVertices(float3& p0, float3& p1, float3& p2) const
    {
        p0 = m_mesh.m_p[m_mesh.m_index[m_index + 0].x];
        p1 = m_mesh.m_p[m_mesh.m_index[m_index + 1].x];
        p2 = m_mesh.m_p[m_mesh.m_index[m_index + 2].x];
    }

    HOST_DEVICE
    AABB worldBound() const
    {
        float3 p0, p1, p2;
        getVertices(p0, p1, p2);
        return Union(AABB(p0, p1), p2);
    }

    HOST_DEVICE
    void sample(
        LightSample* record,
        const float2 s) const
    {
        float3 p0, p1, p2;
        getVertices(p0, p1, p2);
        float2 uv = UniformSampleTriangle(s);
        record->m_p = p0 * (1 - uv.x - uv.y) + p1 * uv.x + p2 * uv.y;

        float3 n = cross(p1 - p0, p2 - p0);
        record->m_normal_g = normalize(n);
        if (m_mesh.m_index[m_index].z != -1)
        {
            float3 n0, n1, n2;
            getNormals(n0, n1, n2);
            record->m_normal_s = normalize(n0 * (1 - uv.x - uv.y) + n1 * uv.x + n2 * uv.y);
        }
        else
        {
            record->m_normal_s = record->m_normal_g;
        }
        //record->m_normal_s = record->m_normal_g;

        record->m_pdf = 2.f / length(n);
    }

    HOST_DEVICE
    void pdf(LightSample& record) const
    {
        float3 p0, p1, p2;
        getVertices(p0, p1, p2);
        float2 uv = record.m_uv;
        record.m_p = p0 * (1 - uv.x - uv.y) + p1 * uv.x + p2 * uv.y;

        float3 n = cross(p1 - p0, p2 - p0);
        record.m_normal_g = normalize(n);
        if (m_mesh.m_index[m_index].z != -1)
        {
            float3 n0, n1, n2;
            getNormals(n0, n1, n2);
            record.m_normal_s = normalize(n0 * (1 - uv.x - uv.y) + n1 * uv.x + n2 * uv.y);
        }
        else
        {
            record.m_normal_s = record.m_normal_g;
        }
        //record->m_normal_s = record->m_normal_g;

        record.m_pdf = 2.f / length(n);
    }

    HOST_DEVICE
    float area() const
    {
        float3 p0, p1, p2;
        getVertices(p0, p1, p2);
        float ret = length(cross(p1 - p0, p2 - p0)) * 0.5f;
        return ret;
    }

    HOST_DEVICE
    bool isLight() const { return m_mesh.m_material.isEmission(); }
    HOST_DEVICE
    Spectrum Le() const { return m_mesh.m_material.m_emission; }

    TriangleMeshView m_mesh;
    uint32 m_index;
};