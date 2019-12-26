#pragma once
#ifndef __TRIANGLE_H
#define __TRIANGLE_H

#include "renderer/core/fwd.h"
#include "renderer/core/transform.h"
#include "renderer/core/parameterset.h"
#include "renderer/core/interaction.h"
#include "renderer/core/sampling.h"


/**
 * counter clock-wise is the normal direction
 */
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
        const Ray& ray) const;

    bool IntersectP(
        const Ray& ray,
        Float* tHit,
        Interaction* interaction) const;

    Interaction Sample(
        Float* pdf, 
        unsigned int& seed) const;

    Point3f Centroid() const;
    Float Area() const;
    Bounds3f WorldBounds() const;



    TriangleMesh* m_triangleMeshPtr;
    int m_index;
    int m_triangleMeshID;
};

std::vector<std::shared_ptr<Triangle>>
CreateTriangleMeshShape(
    const ParameterSet& params,
    Transform objToWorld,
    Transform worldToObj);

std::vector<std::shared_ptr<Triangle>>
CreatePLYMeshShape(
    const ParameterSet& params,
    const Transform& o2w,
    const Transform& w2o);

std::vector<std::shared_ptr<Triangle>>
CreateSphereShape(
    const ParameterSet& params,
    Transform objToWorld,
    Transform worldToObj);


/*
 * Moller-Trumbore algorithm
 */
inline __device__ __host__
bool Triangle::Intersect(const Ray& ray) const
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
    if (std::fabs(det) < Epsilon) {
        return false;
    }
    Float invDet = 1 / det;
    Vector3f T = ray.o - p0;
    Float u = Dot(P, T) * invDet;
    if (u < 0 || u > 1) {
        return false;
    }
    Vector3f Q = Cross(T, E1);
    Float v = Dot(Q, D) * invDet;
    if (v < 0 || u + v > 1) {
        return false;
    }
    Float t = Dot(Q, E2) * invDet;
    if (t < Epsilon || t > ray.tMax) {        
        return false;
    }
    return true;
}

/*
 * Moller-Trumbore algorithm
 */
inline __device__ __host__
bool Triangle::IntersectP(const Ray& ray, Float* tHit, Interaction* interaction) const
{
    int* indices = &m_triangleMeshPtr->m_indices[m_index * 3];
    const Point3f &p0 = m_triangleMeshPtr->m_P[indices[0]];
    const Point3f &p1 = m_triangleMeshPtr->m_P[indices[1]];
    const Point3f &p2 = m_triangleMeshPtr->m_P[indices[2]];

    const Vector3f& D = ray.d;
    Vector3f E1 = p1 - p0;
    Vector3f E2 = p2 - p0;
    Vector3f P = Cross(D, E2);
    Float det = Dot(P, E1);
    if (std::fabs(det) < Epsilon) {
        return false;
    }
    Float invDet = 1 / det;
    Vector3f T = ray.o - p0;
    Float u = Dot(P, T) * invDet;
    if (u < 0 || u > 1) {
        return false;
    }
    Vector3f Q = Cross(T, E1);
    Float v = Dot(Q, D) * invDet;
    if (v < 0 || u + v > 1) {
        return false;
    }
    Float t = Dot(Q, E2) * invDet;
    if (t < Epsilon || t > ray.tMax) {
        return false;
    }
    *tHit = t;
    interaction->m_wo = -ray.d;
    interaction->m_p = ray(t);
    interaction->m_geometryN = Normal3f(Normalize(Cross(E1, E2)));
    if (!m_triangleMeshPtr->m_N) {
        interaction->m_shadingN = interaction->m_geometryN;
    }
    else {
        const Normal3f& n0 = m_triangleMeshPtr->m_N[indices[0]];
        const Normal3f& n1 = m_triangleMeshPtr->m_N[indices[1]];
        const Normal3f& n2 = m_triangleMeshPtr->m_N[indices[2]];
        interaction->m_shadingN = Normalize(n0 * (1 - u - v) + n1 * u + n2 * v);
    }

    if (m_triangleMeshPtr->m_UV) {
        const Point2f& uv0 = m_triangleMeshPtr->m_UV[indices[0]];
        const Point2f& uv1 = m_triangleMeshPtr->m_UV[indices[1]];
        const Point2f& uv2 = m_triangleMeshPtr->m_UV[indices[2]];
        interaction->m_uv = uv0 * (1 - u - v) + uv1 * u + uv2 * v;
    }
    return true;
}

inline __device__ __host__
Interaction Triangle::Sample(Float* pdf, unsigned int& seed) const
{
    Point2f u = UniformSampleTriangle(seed);
    int* indices = &m_triangleMeshPtr->m_indices[m_index * 3];
    const Point3f& p0 = m_triangleMeshPtr->m_P[indices[0]];
    const Point3f& p1 = m_triangleMeshPtr->m_P[indices[1]];
    const Point3f& p2 = m_triangleMeshPtr->m_P[indices[2]];
    Interaction inter;
    inter.m_p = p0 * (1 - u.x - u.y) + p1 * u.x + p2 * u.y;
    inter.m_geometryN = Normal3f(Normalize(Cross(p1 - p0, p2 - p0)));
    if (!m_triangleMeshPtr->m_N) {
        inter.m_shadingN = inter.m_geometryN;
    }
    else {
        const Normal3f& n0 = m_triangleMeshPtr->m_N[indices[0]];
        const Normal3f& n1 = m_triangleMeshPtr->m_N[indices[1]];
        const Normal3f& n2 = m_triangleMeshPtr->m_N[indices[2]];
        inter.m_shadingN = Normalize(n0 * (1 - u.x - u.y) + n1 * u.x + n2 * u.y);
    }    
    *pdf = 1 / Area();
    return inter;
}

inline __device__ __host__
Point3f Triangle::Centroid() const
{
    int* indices = &m_triangleMeshPtr->m_indices[m_index * 3];
    const Point3f& p0 = m_triangleMeshPtr->m_P[indices[0]];
    const Point3f& p1 = m_triangleMeshPtr->m_P[indices[1]];
    const Point3f& p2 = m_triangleMeshPtr->m_P[indices[2]];
    return (p0 + p1 + p2) / 3;
}

inline __device__ __host__
Float Triangle::Area() const
{
    int* indices = &m_triangleMeshPtr->m_indices[m_index * 3];
    const Point3f& p0 = m_triangleMeshPtr->m_P[indices[0]];
    const Point3f& p1 = m_triangleMeshPtr->m_P[indices[1]];
    const Point3f& p2 = m_triangleMeshPtr->m_P[indices[2]];
    return 0.5f * Cross(p1 - p0, p2 - p0).Length();
}

inline __device__ __host__
Bounds3f Triangle::WorldBounds() const
{
    int* indices = &m_triangleMeshPtr->m_indices[m_index * 3];
    const Point3f& p0 = m_triangleMeshPtr->m_P[indices[0]];
    const Point3f& p1 = m_triangleMeshPtr->m_P[indices[1]];
    const Point3f& p2 = m_triangleMeshPtr->m_P[indices[2]];
    return Union(Bounds3f(p0, p1), p2);
}


#endif // !__TRIANGLEMESH_H
