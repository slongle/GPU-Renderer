#pragma once

#include <device_launch_parameters.h>

#include "renderer/triangle.h"
#include "renderer/camera.h"
#include "renderer/ray.h"
#include "renderer/sampling.h"
#include "renderer/rayqueue.h"
#include "renderer/scene.h"
#include "renderer/vertex.h"

#define Epsilon 1e-4f

struct PTContext
{
    uint32 m_bounce;
    uint32 m_sample_num;
    RayQueue m_in_queue;
    RayQueue m_shadow_queue;
    RayQueue m_scatter_queue;
};

void alloc_queues(
    const uint32 pixels_num,
    RayQueue& input_queue,
    RayQueue& scatter_queue,
    RayQueue& shadow_queue,
    MemoryArena& arena)
{
    input_queue.m_rays = arena.alloc<Ray>(pixels_num);
    input_queue.m_hits = arena.alloc<Hit>(pixels_num);
    input_queue.m_weight = arena.alloc<Spectrum>(pixels_num);
    input_queue.m_specular = arena.alloc<bool>(pixels_num);
    input_queue.m_idx = arena.alloc<uint32>(pixels_num);
    input_queue.m_seed = arena.alloc<uint32>(pixels_num);
    input_queue.m_size = arena.alloc<uint32>(1);

    scatter_queue.m_rays = arena.alloc<Ray>(pixels_num);
    scatter_queue.m_hits = arena.alloc<Hit>(pixels_num);
    scatter_queue.m_weight = arena.alloc<Spectrum>(pixels_num);
    scatter_queue.m_specular = arena.alloc<bool>(pixels_num);
    scatter_queue.m_idx = arena.alloc<uint32>(pixels_num);
    scatter_queue.m_seed = arena.alloc<uint32>(pixels_num);
    scatter_queue.m_size = arena.alloc<uint32>(1);

    shadow_queue.m_rays = arena.alloc<Ray>(pixels_num);
    shadow_queue.m_hits = arena.alloc<Hit>(pixels_num);
    shadow_queue.m_weight = arena.alloc<Spectrum>(pixels_num);
    shadow_queue.m_specular = arena.alloc<bool>(pixels_num);
    shadow_queue.m_idx = arena.alloc<uint32>(pixels_num);
    shadow_queue.m_seed = arena.alloc<uint32>(pixels_num);
    shadow_queue.m_size = arena.alloc<uint32>(1);
}

__device__
Ray generate_primary_ray(
    PTContext context,
    SceneView scene,
    FrameBufferView frame_buffer,
    uint2 pixel,
    uint32 idx)
{
    float jitter_x = NextRandom(context.m_in_queue.m_seed[idx]);
    float jitter_y = NextRandom(context.m_in_queue.m_seed[idx]);

    float x = pixel.x + jitter_x;
    float y = scene.m_camera.m_free ? pixel.y + jitter_y : frame_buffer.m_resolution_y - 1 - pixel.y + jitter_y;
    Ray ray = scene.m_camera.generateRay(x, y);

    return ray;
}

__global__ 
void generate_primary_rays_kernel(
    PTContext context,
    SceneView scene,
    FrameBufferView frame_buffer)
{
    const uint2 pixel = make_uint2(
        threadIdx.x + blockIdx.x * blockDim.x,
        threadIdx.y + blockIdx.y * blockDim.y);
    if (pixel.x >= frame_buffer.m_resolution_x || pixel.y >= frame_buffer.m_resolution_y)
    {
        return;
    }

    const uint32 idx = frame_buffer.getIdx(pixel);
    context.m_in_queue.m_seed[idx] = InitRandom(idx, context.m_sample_num);
    const Ray ray = generate_primary_ray(context, scene, frame_buffer, pixel, idx);

    context.m_in_queue.m_rays[idx] = ray;
    context.m_in_queue.m_weight[idx] = Spectrum(1.f);
    context.m_in_queue.m_specular[idx] = false;
    context.m_in_queue.m_idx[idx] = idx;

    if (idx == 0)
    {
        *context.m_in_queue.m_size = frame_buffer.m_resolution_x * frame_buffer.m_resolution_y;
    }
}

void generate_primary_rays(
    PTContext context, 
    SceneView scene, 
    FrameBufferView frame_buffer) 
{
    dim3 block_size(32, 32);
    dim3 grid_size(
        divideRoundInf(frame_buffer.m_resolution_x, block_size.x), 
        divideRoundInf(frame_buffer.m_resolution_y, block_size.y));
    
    generate_primary_rays_kernel << <grid_size, block_size >> > (context, scene, frame_buffer);
}

__device__
bool triangle_intersect(const Triangle* triangle, const Ray ray, Hit* hit) {
    float3 p0, p1, p2;
    triangle->getVertices(p0, p1, p2);
    float3 D = ray.d;
    float3 E1 = p1 - p0;
    float3 E2 = p2 - p0;
    float3 P = cross(D, E2);
    double det = dot(P, E1);


    if (det == 0) {//(det > -Epsilon && det < Epsilon) {        
        return false;
    }
    double invDet = 1.0 / det;
    float3 T = ray.o - p0;
    double u = dot(P, T) * invDet;
    if (u < 0.0 || u > 1.0) {
        return false;
    }
    float3 Q = cross(T, E1);
    double v = dot(Q, D) * invDet;
    if (v < 0.0 || u + v > 1.0) {
        return false;
    }
    double t = dot(Q, E2) * invDet;
    if (t < ray.tMin || t > ray.tMax) {
        return false;
    }
    hit->t = t;
    hit->u = u;
    hit->v = v;
    return true;   
}

__device__
bool aabb_intersect(const AABB& box, const Ray& ray, const float3& inv_dir, const int3& dir_is_neg)
{
    float tMin = (box[dir_is_neg.x].x - ray.o.x) * inv_dir.x;
    float tMax = (box[1 - dir_is_neg.x].x - ray.o.x) * inv_dir.x;
    float tyMin = (box[dir_is_neg.y].y - ray.o.y) * inv_dir.y;
    float tyMax = (box[1 - dir_is_neg.y].y - ray.o.y) * inv_dir.y;

    if (tMin > tyMax || tyMin > tMax) return false;
    tMin = max(tMin, tyMin);
    tMax = min(tMax, tyMax);
    
    float tzMin = (box[dir_is_neg.z].z - ray.o.z) * inv_dir.z;
    float tzMax = (box[1 - dir_is_neg.z].z - ray.o.z) * inv_dir.z;

    if (tMin > tzMax || tzMin > tMax) return false;
    tMin = max(tMin, tzMin);
    tMax = min(tMax, tzMax);
    return (tMin < ray.tMax) && (tMax > ray.tMin);
}

__global__
void trace_kernel(SceneView scene, uint32 size, Ray* rays, Hit* hits)
{
    uint32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size)
    {
        return;
    }
    

    Ray ray = rays[idx];    
    Hit hit;
    hit.triangle_id = -1;

    float3 inv_dir = { 1 / ray.d.x, 1 / ray.d.y, 1 / ray.d.z };
    int3 dir_is_neg = { inv_dir.x < 0, inv_dir.y < 0, inv_dir.z < 0 };

    int currentNodeIndex = 0, toVisitOffset = 0;
    int nodesToVisit[64];
    while (true) {
        BVHLinearNode* node = &scene.m_bvh[currentNodeIndex];
        if (aabb_intersect(node->m_box, ray, inv_dir, dir_is_neg)) {
            if (node->m_leaf) {
                // Leaf node
                if (triangle_intersect(scene.m_triangles + node->m_tri_idx, ray, &hit))
                {
                    ray.tMax = hit.t;
                    hit.triangle_id = node->m_tri_idx;
                }
                if (toVisitOffset == 0) break;
                currentNodeIndex = nodesToVisit[--toVisitOffset];
            }
            else {
                // Interior node
                nodesToVisit[toVisitOffset++] = currentNodeIndex + 1;
                currentNodeIndex = node->m_right_child_idx;
            }
        }
        else {
            if (toVisitOffset == 0) break;
            currentNodeIndex = nodesToVisit[--toVisitOffset];
        }
    }
    hits[idx] = hit;        
    
    /*
    Ray ray = rays[idx];
    Hit hit;
    hit.triangle_id = -1;
    for (uint32 i = 0; i < scene.m_triangles_num; i++)
    {
        if (triangle_intersect(scene.m_triangles + i, ray, &hit))
        {
            ray.tMax = hit.t;
            hit.triangle_id = i;
        }
    }
    hits[idx] = hit;
    */
}

void trace(const SceneView& scene, uint32 size, Ray* rays, Hit* hits)
{
    dim3 block_size(32);
    dim3 grid_size(divideRoundInf(size, block_size.x));

    trace_kernel << <grid_size, block_size >> > (scene, size, rays, hits);
}

__global__
void trace_shadow_kernel(SceneView scene, uint32 size, Ray* rays, Hit* hits)
{
    uint32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size)
    {
        return;
    }

    
    Ray ray = rays[idx];
    Hit hit;
    hit.triangle_id = -1;

    float3 inv_dir = { 1 / ray.d.x, 1 / ray.d.y, 1 / ray.d.z };
    int3 dir_is_neg = { inv_dir.x < 0, inv_dir.y < 0, inv_dir.z < 0 };

    int currentNodeIndex = 0, toVisitOffset = 0;
    int nodesToVisit[64];
    while (true) {
        BVHLinearNode* node = &scene.m_bvh[currentNodeIndex];
        if (aabb_intersect(node->m_box, ray, inv_dir, dir_is_neg)) {
            if (node->m_leaf) {
                // Leaf node
                if (triangle_intersect(scene.m_triangles + node->m_tri_idx, ray, &hit))
                {
                    ray.tMax = hit.t;
                    hit.triangle_id = node->m_tri_idx;
                    break;
                }
                if (toVisitOffset == 0) break;
                currentNodeIndex = nodesToVisit[--toVisitOffset];
            }
            else {
                // Interior node
                nodesToVisit[toVisitOffset++] = currentNodeIndex + 1;
                currentNodeIndex = node->m_right_child_idx;
            }
        }
        else {
            if (toVisitOffset == 0) break;
            currentNodeIndex = nodesToVisit[--toVisitOffset];
        }
    }
    hits[idx] = hit;
    

    /*
    Ray ray = rays[idx];
    Hit hit;
    hit.triangle_id = -1;
    for (uint32 i = 0; i < scene.m_triangles_num; i++)
    {
        if (triangle_intersect(scene.m_triangles + i, ray, &hit))
        {
            ray.tMax = hit.t;
            hit.triangle_id = i;
            break;
        }
    }
    hits[idx] = hit;
    */
}

void trace_shadow(const SceneView& scene, uint32 size, Ray* rays, Hit* hits)
{
    if (size == 0)
    {
        return;
    }

    dim3 block_size(32);
    dim3 grid_size(divideRoundInf(size, block_size.x));

    trace_shadow_kernel << <grid_size, block_size >> > (scene, size, rays, hits);
}

__device__
void sample_lights(
    SceneView scene, 
    LightSample* record, 
    const float3 s) 
{
    uint32 light_num = scene.m_lights_num;
    uint32 light_id = min(uint32(s.x * light_num), light_num - 1);
    Triangle& light = scene.m_lights[light_id];

    record->m_light_id = light_id;
    light.sample(record, make_float2(s.y, s.z));
    record->m_pdf /= light_num;
}

__global__ 
void shade_hit_kernel(
    uint32 in_queue_size,
    PTContext context,
    SceneView scene,
    FrameBufferView frame_buffer)
{
    uint32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= in_queue_size)
    {
        return;
    }

    const uint32 pixel_idx = context.m_in_queue.m_idx[idx];
    const Ray ray = context.m_in_queue.m_rays[idx];
    const Hit hit = context.m_in_queue.m_hits[idx];
    const Spectrum throughput = context.m_in_queue.m_weight[idx];
    const bool specular = context.m_in_queue.m_specular[idx];
    uint32& seed = context.m_in_queue.m_seed[idx];

    float samples[8];
    for (uint32 i = 0; i < 8; ++i)
        samples[i] = NextRandom(seed);

    // Hit
    if (hit.triangle_id != -1)
    {
        const Triangle& triangle = scene.m_triangles[hit.triangle_id];
        Vertex vertex;
        vertex.setup(ray, hit, scene);
       
        //frame_buffer.addRadiance(pixel_idx, Spectrum(vertex.m_normal_g));
        //return;

        if (context.m_bounce == 0 || specular)
        {
            if (triangle.m_mesh.m_material.isEmission())
            {
                Spectrum f_light = dot(vertex.m_wo, vertex.m_normal_s) > 0 ?
                    triangle.m_mesh.m_material.m_emission : Spectrum(0.f);
                frame_buffer.addRadiance(pixel_idx, throughput * f_light);
            }
        }        

        // Next Event Estimate
        if (!vertex.m_bsdf.isDelta() && scene.m_lights_num > 0)
        {
            LightSample light_record;            
            sample_lights(scene, &light_record, make_float3(samples[0], samples[1], samples[2]));
            
            float3 wi = light_record.m_p - vertex.m_p;
            float d2 = square_length(wi);
            float d = sqrtf(d2);
            wi /= d;

            light_record.m_pdf *= d2 / fabsf(dot(wi, light_record.m_normal_s));

            Spectrum f_light = dot(-wi, light_record.m_normal_s) > 0 ?
                scene.m_lights[light_record.m_light_id].m_mesh.m_material.m_emission : Spectrum(0.f);

            BSDFSample bsdf_record;
            bsdf_record.m_wo = vertex.m_wo;
            bsdf_record.m_wi = wi;
            vertex.m_bsdf.eval(bsdf_record);
            Spectrum f_bsdf = bsdf_record.m_f; 

            Spectrum out_weight = throughput * f_light * f_bsdf / light_record.m_pdf;

            if (!isBlack(out_weight))
            {
                Ray shadow_ray;
                shadow_ray.o = vertex.m_p - ray.d * 1.0e-4f;
                shadow_ray.d = light_record.m_p - shadow_ray.o;
                shadow_ray.tMin = 0.f;
                shadow_ray.tMax = 0.9999f;
                //float dist = length(light_record.m_p - vertex.m_p);
                //shadow_ray.o = vertex.m_p;
                //shadow_ray.d = normalize(light_record.m_p - vertex.m_p);
                //shadow_ray.tMin = 1e-4f;
                //shadow_ray.tMax = dist * (1 - 1e-3f);

                context.m_shadow_queue.append(shadow_ray, out_weight, true, pixel_idx);
            }
        }
       
        // Scatter
        {
            BSDFSample bsdf_record;
            bsdf_record.m_wo = vertex.m_wo;
            vertex.m_bsdf.sample(make_float2(samples[5], samples[6]), bsdf_record);

            Spectrum out_weight = bsdf_record.m_pdf == 0.f ? Spectrum(0.f) : throughput * bsdf_record.m_f / bsdf_record.m_pdf;
            bool out_specular = bsdf_record.m_specular;

            if (fmaxf(out_weight) < 1 && context.m_bounce > 3) {
                float q = fmaxf(0.05f, 1 - fmaxf(throughput));
                if (samples[7] < q)
                {
                    out_weight = Spectrum(0);
                }
                out_weight /= 1 - q;
            }

            if (!isBlack(out_weight))
            {                
                Ray scatter_ray;
                scatter_ray.o = vertex.m_p;
                scatter_ray.d = bsdf_record.m_wi;
                scatter_ray.tMin = 1e-4f;
                scatter_ray.tMax = 1e18f;

                context.m_scatter_queue.append(scatter_ray, out_weight, out_specular, pixel_idx);
            }            
        }
    }
    else
    {
        // Environment Light
        //frame_buffer.addRadiance(pixel_idx, throughput);
        if (scene.m_environment_light.m_has)
        {
            Spectrum le = scene.m_environment_light.Le(ray);
            Spectrum l = throughput * le;
            frame_buffer.addRadiance(pixel_idx, l);
        }
    }
}

void shade_hit(
    uint32 in_queue_size,
    PTContext context,
    SceneView scene,
    FrameBufferView frame_buffer)
{
    dim3 block_size(32);
    dim3 grid_size(divideRoundInf(in_queue_size, block_size.x));

    shade_hit_kernel << <grid_size, block_size >> > (in_queue_size, context, scene, frame_buffer);
}

__global__
void accumulate_kernel(
    uint32 shadow_queue_size,
    PTContext context,
    SceneView scene,
    FrameBufferView frame_buffer)
{
    uint32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= shadow_queue_size)
    {
        return;
    }

    const Hit& hit = context.m_shadow_queue.m_hits[idx];
    const uint32& pixel_idx = context.m_shadow_queue.m_idx[idx];
    const Spectrum& L = context.m_shadow_queue.m_weight[idx];

    if (hit.triangle_id == -1)
    {
        frame_buffer.addRadiance(pixel_idx, L);
    }
}

void accumulate(
    uint32 shadow_queue_size,
    PTContext context,
    SceneView scene,
    FrameBufferView frame_buffer)
{
    if (shadow_queue_size == 0)
    {
        return;
    }

    dim3 block_size(32);
    dim3 grid_size(divideRoundInf(shadow_queue_size, block_size.x));

    accumulate_kernel << <grid_size, block_size >> > (shadow_queue_size, context, scene, frame_buffer);
}

__global__
void finish_sample_kernel(
    uint32  size,
    FrameBufferView frame_buffer)
{
    uint32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size)
    {
        return;
    }
    frame_buffer.addSampleNum(idx);
}

void finish_sample(
    FrameBufferView frame_buffer)
{
    uint32 size = frame_buffer.m_resolution_x * frame_buffer.m_resolution_y;
    dim3 block_size(32);
    dim3 grid_size(divideRoundInf(size, block_size.x));

    finish_sample_kernel << <grid_size, block_size >> > (size, frame_buffer);
}

__global__
void filter_kernel(
    uint32  size,
    uint32* output,
    FrameBufferView frame_buffer)
{
    uint32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size)
    {
        return;
    }

    Spectrum color = frame_buffer.m_buffer[idx] / frame_buffer.m_sample_num[idx];
    uint8* ptr_uc = (uint8*)(output + idx);
    ptr_uc[0] = (uint8)clamp(255.f * GammaCorrect(color.r) + 0.5f, 0.f, 255.f);
    ptr_uc[1] = (uint8)clamp(255.f * GammaCorrect(color.g) + 0.5f, 0.f, 255.f);
    ptr_uc[2] = (uint8)clamp(255.f * GammaCorrect(color.b) + 0.5f, 0.f, 255.f);
    ptr_uc[3] = (uint8)clamp(255.f * GammaCorrect(1) + 0.5f, 0.f, 255.f);
}

void filter(
    uint32* output,
    FrameBufferView frame_buffer)
{
    uint32 size = frame_buffer.m_resolution_x * frame_buffer.m_resolution_y;
    dim3 block_size(32);
    dim3 grid_size(divideRoundInf(size, block_size.x));

    filter_kernel << <grid_size, block_size >> > (size, output, frame_buffer);
}
