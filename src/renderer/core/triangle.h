#pragma once
#ifndef __TRIANGLE_H
#define __TRIANGLE_H

#include "renderer/core/fwd.h"
#include "renderer/core/transform.h"
#include "renderer/core/parameterset.h"

#include <thrust/device_vector.h>

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

    TriangleMesh* m_triangleMeshPtr;
    int m_index;
    int m_triangleMeshID;
};

std::vector<std::shared_ptr<Triangle>>
CreateTriangleMeshShape(
    const ParameterSet& params, 
    Transform objToWorld, 
    Transform worldToObj);

#endif // !__TRIANGLEMESH_H
