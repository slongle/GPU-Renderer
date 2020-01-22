#pragma once

#include <device_launch_parameters.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>

#include <algorithm>

#include "renderer/aabb.h"
#include "renderer/triangle.h"

inline 
uint32 LeftShift3(uint32 x) {    
    assert(x <= 1024);
    if (x == (1 << 10)) --x;
    x = (x | (x << 16)) & 0x30000ff;
    // x = ---- --98 ---- ---- ---- ---- 7654 3210
    x = (x | (x << 8)) & 0x300f00f;
    // x = ---- --98 ---- ---- 7654 ---- ---- 3210
    x = (x | (x << 4)) & 0x30c30c3;
    // x = ---- --98 ---- 76-- --54 ---- 32-- --10
    x = (x | (x << 2)) & 0x9249249;
    // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
    return x;
}

inline 
uint32_t EncodeMorton3(const float3& v) {
    //printf("%f %f %f\n", v.x, v.y, v.z);
    assert(v.x >= 0 && v.y >= 0 && v.z >= 0);
    return (LeftShift3(v.z) << 2) | (LeftShift3(v.y) << 1) | LeftShift3(v.x);
}

struct BVHLinearNode
{
    AABB m_box;
    union
    {
        uint32 m_tri_idx;
        uint32 m_right_child_idx;
    };
    uint8 m_leaf;
};

struct BVHBuildNode
{
    AABB m_box;
    uint32 m_tri_idx;
    BVHBuildNode* m_left, * m_right;
};

struct BVHInfo
{
    AABB m_box;
    float3 m_centroid;
};

struct MortonInfo
{
    uint32 m_tri_idx;
    uint32 m_morton;
};

__global__ inline
void init_triangle_information_kernel(
    uint32 tri_num,
    Triangle* triangles,
    BVHInfo* tri_info)
{
    uint32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= tri_num)
    {
        return;
    }

    tri_info[idx].m_box = triangles[idx].worldBound();
    tri_info[idx].m_centroid = tri_info[idx].m_box.centroid();
}

inline
void init_triangle_information(
    uint32 tri_num,
    Triangle* triangles,
    BVHInfo* tri_info)
{
    dim3 block_size(16, 16);
    dim3 grid_size(
        divideRoundInf(tri_num, block_size.x),
        divideRoundInf(tri_num, block_size.y));
    //init_triangle_information_kernel << <grid_size, block_size >> > (tri_num, triangles, tri_info);
}

__global__ inline 
void calc_union_bound_kernel(
    uint32 tri_num,
    BVHInfo* tri_info,
    AABB* tmp)
{
    uint32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= tri_num)
    {
        return;
    }
    
    tmp[idx] = AABB(tri_info[idx].m_centroid);


}

inline
void calc_union_bound(
    uint32 tri_num,
    BVHInfo* tri_info,
    AABB* tmp)
{
    dim3 block_size(16, 16);
    dim3 grid_size(
        divideRoundInf(tri_num, block_size.x),
        divideRoundInf(tri_num, block_size.y));
    //calc_union_bound_kernel << <grid_size, block_size >> > (tri_num, tri_info, tmp);
}

inline uint32
flatten_nodes(
    BVHBuildNode* node,
    BVHLinearNode* buffer,
    uint32& linear_nodes_num)
{
    uint32 idx = linear_nodes_num ++;
    buffer[idx].m_box = node->m_box;
    if (node->m_left == nullptr && node->m_right == nullptr)
    {
        buffer[idx].m_tri_idx = node->m_tri_idx;
        buffer[idx].m_leaf = 1;
    }
    else 
    {
        flatten_nodes(node->m_left, buffer, linear_nodes_num);
        buffer[idx].m_right_child_idx = flatten_nodes(node->m_right, buffer, linear_nodes_num);
        buffer[idx].m_leaf = 0;
    }
    return idx;
}

inline BVHBuildNode*
recursive_build(
    const std::vector<MortonInfo>& mortons,
    const std::vector<Triangle>& triangles,
    const uint32 l, const uint32 r,
    std::vector<BVHBuildNode>& build_nodes,
    uint32& build_nodes_num)
{
    if (l + 1 == r)
    {
        BVHBuildNode* node = &build_nodes[build_nodes_num ++];
        node->m_left = nullptr;
        node->m_right = nullptr;
        node->m_tri_idx = mortons[l].m_tri_idx;
        node->m_box = triangles[mortons[l].m_tri_idx].worldBound();
        return node;
    }

    uint32 split_idx = l;
    if (mortons[l].m_morton == mortons[r - 1].m_morton)
    {
        split_idx = ((l + r) >> 1) - 1;
    }
    else
    {
        for (int i = l; i < r - 1; i++) {
            uint32 xor_now = mortons[i].m_morton ^ mortons[i + 1].m_morton;
            uint32 xor_split = mortons[split_idx].m_morton ^ mortons[split_idx + 1].m_morton;
            if (xor_now > xor_split)
            {
                split_idx = i;
            }
        }
    }
    BVHBuildNode* l_child = recursive_build(mortons, triangles, l, split_idx + 1, build_nodes, build_nodes_num);
    BVHBuildNode* r_child = recursive_build(mortons, triangles, split_idx + 1, r, build_nodes, build_nodes_num);
    BVHBuildNode* node = &build_nodes[build_nodes_num++];
    node->m_left = l_child;
    node->m_right = r_child;
    node->m_box = Union(l_child->m_box, r_child->m_box);
    return node;
}

inline std::vector<BVHLinearNode>
LBVH_build(
    const std::vector<Triangle>& triangles
)
{
    uint32 tri_num = triangles.size();
    AABB box;
    for (int i = 0; i < tri_num; i++) {
        box = Union(box, triangles[i].worldBound().centroid());
    }

    std::vector<MortonInfo> mortons(tri_num);
    for (int i = 0; i < tri_num; i++) {
        mortons[i].m_tri_idx = i;
        mortons[i].m_morton = EncodeMorton3((triangles[i].worldBound().centroid() - box.m_min) / (box.m_max - box.m_min) * 1024);
    }

    std::sort(mortons.begin(), mortons.end(), [](const MortonInfo& m1, const MortonInfo& m2) {return m1.m_morton < m2.m_morton; });

    /*for (int i = 0; i < tri_num; i++) {
        std::cout << mortons[i].m_morton << std::endl;
    }*/

    std::vector<BVHBuildNode> build_nodes(2 * tri_num - 1);
    uint32 build_nodes_num = 0;
    BVHBuildNode* root = recursive_build(mortons, triangles, 0, tri_num, build_nodes, build_nodes_num);

    std::vector<BVHLinearNode> bvh_buffer(2 * tri_num - 1);
    uint32 linear_nodes_num = 0;
    flatten_nodes(root, bvh_buffer.data(), linear_nodes_num);

    return bvh_buffer;
}