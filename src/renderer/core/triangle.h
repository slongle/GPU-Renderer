#pragma once
#ifndef __TRIANGLE_H
#define __TRIANGLE_H

#include "renderer/core/fwd.h"
#include "renderer/core/transform.h"
#include "renderer/core/parameterset.h"
#include "renderer/core/interaction.h"

class TriangleMesh{
public:
    TriangleMesh() {}
    TriangleMesh(
        Transform objToWorld,
        std::vector<int>& indices,
        std::vector<Point3f>& p,
        std::vector<Normal3f>& n,
        std::vector<float>& uv);

    ~TriangleMesh() {
        /*
        delete[] m_indices;
        delete[] m_P;
        delete[] m_N;
        delete[] m_UV;
        */
    }

    int m_triangleNum;
    int m_vertexNum;
    int* m_indices = nullptr;
    Point3f* m_P = nullptr;
    Normal3f* m_N = nullptr;
    Point2f* m_UV = nullptr;
};

class Triangle {
public:
    Triangle(
        TriangleMesh* triangleMeshPtr,
        int index);

    bool Intersect(
        const Ray& ray, 
        Float* tHit, 
        Interaction* interaction) const;

    TriangleMesh* m_triangleMeshPtr;
    int m_index;
    int m_triangleMeshID;
};

inline
std::vector<std::shared_ptr<Triangle>>
CreateTriangleMeshShape(
    const ParameterSet& params,
    Transform objToWorld,
    Transform worldToObj);

inline
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

inline
Triangle::Triangle(
    TriangleMesh* triangleMeshPtr, int index)
    : m_triangleMeshPtr(triangleMeshPtr), m_index(index), m_triangleMeshID(-1) {}

/*
 * Moller-Trumbore algorithm
 */
inline __device__ __host__
bool Triangle::Intersect(const Ray& ray, Float* tHit, Interaction* interaction) const
{
    int* indices = &m_triangleMeshPtr->m_indices[m_index * 3];
    Point3f p0 = m_triangleMeshPtr->m_P[indices[0]];
    Point3f p1 = m_triangleMeshPtr->m_P[indices[1]];
    Point3f p2 = m_triangleMeshPtr->m_P[indices[2]];

    const Vector3f& D = ray.d;
    Vector3f E1 = p1 - p0;
    Vector3f E2 = p2 - p0;
    Vector3f P = Cross(D, E2);
    Float det = Dot(P, E1);
    if (std::fabs(det) < EPSILON) {
        return false;
    }
    Float invDet = 1 / det;
    Vector3f T = ray.o - p0;
    Vector3f Q = Cross(T, E1);
    Float u = Dot(P, T) * invDet;
    if (u < 0 || u>1) {
        return false;
    }
    Float v = Dot(Q, D) * invDet;
    if (v < 0 || u + v > 1) {
        return false;
    }
    Float t = Dot(Q, E2) * invDet;
    if (t < ray.tMax) {
        return false;
    }
    *tHit = t;

    return true;
}

inline
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


#endif // !__TRIANGLEMESH_H
