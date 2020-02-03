#include "triangle.h"

TriangleMeshView::TriangleMeshView(const TriangleMesh* mesh)
    : m_triangle_num(mesh->m_triangle_num),
      m_p(mesh->m_gpu_p.data()), m_p_num(mesh->m_gpu_p.size()),
      m_n(mesh->m_gpu_n.data()), m_n_num(mesh->m_gpu_p.size()),
      m_uv(mesh->m_gpu_uv.data()), m_uv_num(mesh->m_gpu_p.size()),
      m_index(mesh->m_gpu_index.data()), m_index_num(mesh->m_gpu_p.size()),
      m_material(mesh->m_material)
{
}

TriangleMeshView::TriangleMeshView(const TriangleMesh* mesh, bool host)
    : m_triangle_num(mesh->m_triangle_num),
      m_p(mesh->m_cpu_p.data()), m_p_num(mesh->m_cpu_p.size()),
      m_n(mesh->m_cpu_n.data()), m_n_num(mesh->m_cpu_p.size()),
      m_uv(mesh->m_cpu_uv.data()), m_uv_num(mesh->m_cpu_p.size()),
      m_index(mesh->m_cpu_index.data()), m_index_num(mesh->m_cpu_p.size()),
      m_material(mesh->m_material)
{
}

void TriangleMesh::createDeviceData()
{
    m_gpu_p.copyFrom(m_cpu_p.size(), HOST_BUFFER, m_cpu_p.data());
    m_gpu_n.copyFrom(m_cpu_n.size(), HOST_BUFFER, m_cpu_n.data());
    m_gpu_uv.copyFrom(m_cpu_uv.size(), HOST_BUFFER, m_cpu_uv.data());
    m_gpu_index.copyFrom(m_cpu_index.size(), HOST_BUFFER, m_cpu_index.data());
}

Triangle::Triangle(uint32 index, TriangleMeshView mesh)
    : m_mesh(mesh), m_index(index)
{
}
