#include "triangle.h"

TriangleMesh::TriangleMesh(
    Transform objToWorld,
    std::vector<int>& indices,
    std::vector<Point3f>& p,
    std::vector<Normal3f>& n,
    std::vector<float>& uv)
    : m_triangleNum(indices.size() / 3), m_vertexNum(p.size())
{
    m_indices = new int[m_triangleNum * 3];
    for (int i = 0; i < m_triangleNum * 3; i++) {
        m_indices[i] = indices[i];
    }
    m_P = new Point3f[m_vertexNum];
    for (int i = 0; i < m_vertexNum; i++) {
        m_P[i] = objToWorld(p[i]);
    }
    if (n.size() != 0) {
        m_N = new Normal3f[m_vertexNum];
        for (int i = 0; i < m_vertexNum; i++) {
            m_N[i] = objToWorld(n[i]);
        }
    }
    if (uv.size() != 0) {
        m_UV = new Point2f[m_vertexNum];
        for (int i = 0; i < m_vertexNum; i += 2) {            
            m_UV[i] = Point2f(uv[i], uv[i + 1]);
        }
    }
}

Triangle::Triangle(
    TriangleMesh* triangleMeshPtr, int index)
    : m_triangleMeshPtr(triangleMeshPtr), m_index(index), m_triangleMeshID(-1) {}

std::vector<std::shared_ptr<Triangle>> 
CreateTriangleMeshShape(
    const ParameterSet& params, 
    Transform objToWorld, 
    Transform worldToObj)
{
    std::vector<int> indices = params.GetInts("indices");
    std::vector<Point3f> p = params.GetPoints("P");
    std::vector<Normal3f> n = params.GetNormals("N");
    std::vector<Float> uv = params.GetFloats("uv", std::vector<Float>());

    TriangleMesh* triangleMesh = new TriangleMesh(objToWorld, indices, p, n, uv);
    std::vector<std::shared_ptr<Triangle>> triangles;
    int triangleNum = indices.size() / 3;
    for (int i = 0; i < triangleNum; i++) {
        triangles.push_back(std::make_shared<Triangle>(triangleMesh, i));
    }

    return triangles;
}

