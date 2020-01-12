#pragma once
#include "renderer/core/primitive.h"
#include "renderer/core/bvh.h"

/*struct LinearBVHNode {
    Bounds3f bounds;
    union {
        int primitivesOffset; // Leaf      the offset in m_primitives array
        int rightChildOffset; // Interior  the offset in node array
    };
    uint16_t nPrimitives;     // the number of m_primitives in this node
    uint8_t axis;             // SplitAxis   
    uint8_t pad;              // Ensure 32 byte size
};*/

class CUDABVH {
public:
    enum SplitMethod { SAH = 0, Middle, EqualCounts };

    CUDABVH() { }

    bool IntersectP(
        const Ray& ray,
        Interaction* inter,
        const Triangle* triangles) const;
    bool Intersect(
        const Ray& ray,
        const Triangle* triangles) const;

    Primitive* m_primitives = nullptr;
    int m_maxPrimsInNode, m_totalNodes;
    SplitMethod m_splitMethod;
    LinearBVHNode* m_nodes = nullptr;
};

__host__ __device__
bool CUDABVH::IntersectP(
    const Ray& ray,
    Interaction* inter,
    const Triangle* triangles) const
{
    if (!m_nodes) return false;
    bool hit = false;
    Vector3f invDir(1 / ray.d.x, 1 / ray.d.y, 1 / ray.d.z);
    int dirIsNeg[3] = { invDir.x < 0,invDir.y < 0,invDir.z < 0 };

    int currentNodeIndex = 0, toVisitOffset = 0;
    int nodesToVisit[64];
    while (true) {
        LinearBVHNode* node = &m_nodes[currentNodeIndex];
        if (node->bounds.Intersect(ray, invDir, dirIsNeg)) {
            if (node->nPrimitives > 0) {
                // Leaf node
                for (int i = 0; i < node->nPrimitives; i++) {
                    Float tHit;
                    int id = m_primitives[node->primitivesOffset + i].m_shapeID;
                    if (triangles[id].IntersectP(ray, &tHit, inter)) {
                        ray.tMax = tHit;
                        inter->m_primitiveID = id;
                        hit = true;
                    }
                }
                if (toVisitOffset == 0) break;
                currentNodeIndex = nodesToVisit[--toVisitOffset];
            }
            else {
                // Interior node
                if (dirIsNeg[node->axis]) {
                    nodesToVisit[toVisitOffset++] = currentNodeIndex + 1;
                    currentNodeIndex = node->rightChildOffset;
                }
                else {
                    nodesToVisit[toVisitOffset++] = node->rightChildOffset;
                    currentNodeIndex = currentNodeIndex + 1;
                }
            }
        }
        else {
            if (toVisitOffset == 0) break;
            currentNodeIndex = nodesToVisit[--toVisitOffset];
        }
    }
    return hit;
}

__host__ __device__
bool CUDABVH::Intersect(
    const Ray& ray,
    const Triangle* triangles) const
{
    if (!m_nodes) return false;
    Vector3f invDir(1 / ray.d.x, 1 / ray.d.y, 1 / ray.d.z);
    int dirIsNeg[3] = { invDir.x < 0,invDir.y < 0,invDir.z < 0 };

    int currentNodeIndex = 0, toVisitOffset = 0;
    int nodesToVisit[64];
    while (true) {
        LinearBVHNode* node = &m_nodes[currentNodeIndex];
        if (node->bounds.Intersect(ray, invDir, dirIsNeg)) {
            if (node->nPrimitives > 0) {
                // Leaf node
                for (int i = 0; i < node->nPrimitives; i++) {
                    int id = m_primitives[node->primitivesOffset + i].m_shapeID;
                    if (triangles[id].Intersect(ray)) {
                        return true;
                    }
                }
                if (toVisitOffset == 0) break;
                currentNodeIndex = nodesToVisit[--toVisitOffset];
            }
            else {
                // Interior node
                if (dirIsNeg[node->axis]) {
                    nodesToVisit[toVisitOffset++] = currentNodeIndex + 1;
                    currentNodeIndex = node->rightChildOffset;
                }
                else {
                    nodesToVisit[toVisitOffset++] = node->rightChildOffset;
                    currentNodeIndex = currentNodeIndex + 1;
                }
            }
        }
        else {
            if (toVisitOffset == 0) break;
            currentNodeIndex = nodesToVisit[--toVisitOffset];
        }
    }
    return false;
}