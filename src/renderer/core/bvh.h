#pragma once
#ifndef __BVH_H
#define __BVH_H

#include "renderer/core/fwd.h"
#include "renderer/core/geometry.h"
#include "renderer/core/triangle.h"
#include "renderer/core/primitive.h"
#include "renderer/core/memory.h"

/*
#include <thrust/sort.h>

class BVH
{
public:
    BVH() {}
    struct Node
    {
        Node() {}
        int idx;
        int flag;
        Triangle* triangle = 0;
        Node* childA = 0;
        Node* childB = 0;
        Node* parent = 0;
        Bounds3f Box;

    };

    void Build(std::vector<Primitive>& primitives, std::vector<Triangle>& triangles);
    bool InterTest(const Ray& ray, Node* p) const;
    bool InterTestP(const Ray& ray, Interaction* interaction, Node* p) const;
    bool Intersect(const Ray& ray) const;
    bool IntersectP(const Ray& ray, Interaction* interaction) const;

    int m_length;
    Node* m_root;

};

inline __host__ __device__
Vector3f DirectDiv(Vector3f a, Vector3f b)
{
    Float x = (b.x != 0) ? (a.x / b.x) : 0;
    Float y = (b.y != 0) ? (a.y / b.y) : 0;
    Float z = (b.z != 0) ? (a.z / b.z) : 0;
    return Vector3f(x, y, z);
}

inline __host__ __device__
unsigned int Expand(unsigned int v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

inline __host__ __device__
unsigned int Morton(Point3f p, Bounds3f sceneBox)
{
    Vector3f v = DirectDiv(p - sceneBox.pMin, sceneBox.pMax - sceneBox.pMin) * 1023.;
    unsigned int x = Expand((unsigned int)v.x);
    unsigned int y = Expand((unsigned int)v.y);
    unsigned int z = Expand((unsigned int)v.z);
    return (x << 2) + (y << 1) + (z);
}
inline __host__ __device__
Point3f pMin(Point3f a, Point3f b)
{
    return Point3f(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z));
}
inline __host__ __device__
Point3f pMax(Point3f a, Point3f b)
{
    return Point3f(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z));
}


inline __host__ __device__
void BVH::Build(std::vector<Primitive>& primitives, std::vector<Triangle>& triangles)
{
    m_length = primitives.size();
    Node* leafNodes = new Node[m_length];
    Node* innerNodes = new Node[m_length];
    unsigned int* morton = (unsigned int*)malloc(sizeof(unsigned int) * m_length);
    int* level = (int*)malloc(sizeof(int) * m_length);
    int* idx = (int*)malloc(sizeof(int) * m_length);
    Bounds3f sceneBox = Bounds3f(triangles[0].Centroid());

    for (int i = 1; i < m_length; i++)
    {
        sceneBox = Union(sceneBox, Bounds3f(triangles[i].Centroid()));
    }

    for (int i = 0; i < m_length; i++)
    {
        int idx = primitives[i].m_shapeID;
        morton[i] = Morton(triangles[idx].Centroid(), sceneBox);
    }
    for (int i = 0; i < m_length; i++)
    {
        idx[i] = i;
    }

    thrust::sort_by_key(morton, morton + m_length, idx);
    for (int i = 0; i < m_length; i++)
    {
        leafNodes[i].idx = idx[i];
        int triangleID = primitives[idx[i]].m_shapeID;
        Triangle* triangle = &triangles[triangleID];
        leafNodes[i].triangle = triangle;
        int* indices = &triangle->m_triangleMeshPtr->m_indices[triangle->m_index * 3];
        Point3f p0 = triangle->m_triangleMeshPtr->m_P[indices[0]];
        Point3f p1 = triangle->m_triangleMeshPtr->m_P[indices[1]];
        Point3f p2 = triangle->m_triangleMeshPtr->m_P[indices[2]];
        leafNodes[i].Box = Union(Bounds3f(p0, p1), Bounds3f(p2));
        leafNodes[i].flag = 3;
    }
    level[0] = 30;
    for (int i = 1; i < m_length; i++)
    {
        level[i] = 0;

        unsigned int tmp = (morton[i] ^ morton[i - 1]);
        unsigned int k = 1 << 29;
        while (k!=0 && (k & tmp) == 0)
        {
            level[i]++;
            k >>= 1;
        }
    }

    for (int i = 1; i < m_length; i++)
    {
        innerNodes[i].idx = i;
        int j = i - 1;
        int k = 0;
        while ((j > 0) && (level[j] >= level[i]))
        {
            if (level[k] > level[j])
            {
                k = j;
            }
            j--;
        }
        innerNodes[i].childA = &innerNodes[k];
        if (k == 0)
        {
            innerNodes[i].childA = &leafNodes[i - 1];
        }
        innerNodes[i].childA->parent = &innerNodes[i];

        j = i + 1;
        k = 0;
        while ((j < m_length) && (level[j] > level[i]))
        {
            if (level[k] > level[j])
            {
                k = j;
            }
            j++;
        }
        innerNodes[i].childB = &innerNodes[k];
        if (k == 0)
        {
            innerNodes[i].childB = &leafNodes[i];
        }
        innerNodes[i].childB->parent = &innerNodes[i];


    }
    int* flag = (int*)malloc(sizeof(int) * m_length);
    for (int i = 0; i < m_length; i++)
    {
        flag[i] = 0;
        innerNodes[i].flag = 0;
    }
    for (int i = 0; i < m_length; i++)
    {
        Node* p = leafNodes[i].parent;
        while (p != 0)
        {
            flag[p->idx]++;
            if (flag[p->idx] == 1) break;
            p->Box = Union(p->childA->Box, p->childB->Box);
            //			printf("     %d %d  %d\n", p->idx, p->childA->idx, p->childB->idx);
            p = p->parent;
        }
    }
    Node* p = leafNodes[0].parent;
    while (p->parent != 0) {
        p = p->parent;
    }
    m_root = p;
    //	puts("NO");
}

inline __host__ __device__
bool BVH::InterTest(const Ray& ray, Node* p)const
{
    if (p->Box.Intersect(ray)) {
        if (p->flag == 3)
        {
            return p->triangle->Intersect(ray);
        }
        return InterTest(ray, p->childA) || InterTest(ray, p->childB);
    }
    return false;
}


inline __host__ __device__
bool BVH::Intersect(const Ray& ray) const
{
    Node* p = m_root;
    return InterTest(ray, p);
}

inline __host__ __device__
bool BVH::InterTestP(const Ray& ray, Interaction* interaction, Node* p)const
{
    Float tHit;

    if (p->Box.Intersect(ray)) {
        if (p->flag == 3)
        {
            bool hit = p->triangle->IntersectP(ray, &tHit, interaction);
            if (hit) {
                ray.tMax = tHit;
                interaction->m_primitiveID = p->idx;
                //				printf("    %d\n", p->idx);
                return true;
            }
            return false;
        }
        bool ret_hitA = InterTestP(ray, interaction, p->childA);
        bool ret_hitB = InterTestP(ray, interaction, p->childB);
        return ret_hitA || ret_hitB;
    }
    return false;
}

inline __host__ __device__
bool BVH::IntersectP(const Ray& ray, Interaction* interaction) const
{
    Node* p = nullptr;
    p = m_root;
    return InterTestP(ray, interaction, p);
}
*/

struct BVHPrimitiveInfo;
struct BVHBuildNode;
struct LinearBVHNode;

class BVHAccelerator {
public:
    enum SplitMethod { SAH, Middle, EqualCounts };

    BVHAccelerator() {}
    BVHAccelerator(
        std::vector<Primitive>& primitives,
        const std::vector<Triangle>& triangles,
        const SplitMethod& splitMethod = SAH,
        int maxPrimsInNode = 255);

    void Build(
        std::vector<Primitive>& primitives,
        const std::vector<Triangle>& triangles,
        const SplitMethod& splitMethod = SAH,
        int maxPrimsInNode = 255);

    BVHBuildNode* RecursiveBuild(
        MemoryArena& arena,
        std::vector<BVHPrimitiveInfo>& primitiveInfo,
        unsigned int begin, 
        unsigned int end, 
        unsigned int* totalNodes,
        std::vector<Primitive>& orderedPrims);

    int FlattenBVHTree(BVHBuildNode* node, int* offset);

    bool IntersectP(
        const Ray& ray, 
        Interaction* inter, 
        const Triangle* triangles) const;
    bool Intersect(
        const Ray& ray, 
        const Triangle* triangles) const;

    std::vector<Primitive> m_primitives;
    int m_maxPrimsInNode;
    SplitMethod m_splitMethod;
    LinearBVHNode* m_nodes = nullptr;    
};

#endif // __BVH_H 