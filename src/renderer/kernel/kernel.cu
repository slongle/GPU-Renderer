#ifndef _RENDER_KERNEL_CU_
#define _RENDER_KERNEL_CU_

#include "utility/helper_cuda.h"
#include "utility/helper_math.h"
#include "utility/helper_functions.h"

// CUDA Runtime, includes
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <device_launch_parameters.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <driver_functions.h>

#include "renderer/core/transform.h"
#include "renderer/core/renderer.h"

#include "renderer/kernel/cudascene.h"

CUDAScene* hst_scene;
CUDAScene* dev_scene;
Camera* hst_camera;
Camera* dev_camera;
Integrator* hst_integrator;
Integrator* dev_integrator;

extern "C"
void cudaInit(std::shared_ptr<Renderer> renderer) {
    // Move Scene Data
    Scene* scene = &(renderer->m_scene);
    hst_scene = new CUDAScene(scene);  

    // Move TriangleMesh Data
    int triangleMeshNum = hst_scene->m_triangleMeshNum;
    for (int i = 0; i < triangleMeshNum; i++) {
        int triangleNum = scene->m_triangleMeshes[i].m_triangleNum;
        cudaMalloc(&hst_scene->m_triangleMeshes[i].m_indices, 3 * triangleNum * sizeof(int));
        cudaMemcpy(hst_scene->m_triangleMeshes[i].m_indices, scene->m_triangleMeshes[i].m_indices,
            3 * triangleNum * sizeof(int), cudaMemcpyHostToDevice);
        int vertexNum = scene->m_triangleMeshes[i].m_vertexNum;
        cudaMalloc(&hst_scene->m_triangleMeshes[i].m_P, vertexNum * sizeof(Point3f));
        cudaMemcpy(hst_scene->m_triangleMeshes[i].m_P, scene->m_triangleMeshes[i].m_P,
            vertexNum * sizeof(Point3f), cudaMemcpyHostToDevice);

        if (scene->m_triangleMeshes[i].m_N) {
            cudaMalloc(&hst_scene->m_triangleMeshes[i].m_N, vertexNum * sizeof(Normal3f));
            cudaMemcpy(hst_scene->m_triangleMeshes[i].m_N, scene->m_triangleMeshes[i].m_N,
                vertexNum * sizeof(Normal3f), cudaMemcpyHostToDevice);
        }

        if (scene->m_triangleMeshes[i].m_UV) {
            cudaMalloc(&hst_scene->m_triangleMeshes[i].m_UV, vertexNum * sizeof(Point2f));
            cudaMemcpy(hst_scene->m_triangleMeshes[i].m_UV, scene->m_triangleMeshes[i].m_UV,
                vertexNum * sizeof(Point2f), cudaMemcpyHostToDevice);
        }
    }
    TriangleMesh* triangleMeshGPUPtr;
    cudaMalloc(&triangleMeshGPUPtr, triangleMeshNum * sizeof(TriangleMesh));
    cudaMemcpy(triangleMeshGPUPtr, hst_scene->m_triangleMeshes, triangleMeshNum * sizeof(TriangleMesh), cudaMemcpyHostToDevice);
    hst_scene->m_triangleMeshes = triangleMeshGPUPtr;

    // Move Triangle Data
    int triangleNum = scene->m_triangles.size();
    cudaMalloc(&hst_scene->m_triangles, sizeof(Triangle) * triangleNum);
    cudaMemcpy(hst_scene->m_triangles, scene->m_triangles.data(),
        sizeof(Triangle) * triangleNum, cudaMemcpyHostToDevice);

    // Move Material Data
    int materialNum = scene->m_materials.size();
    cudaMalloc(&hst_scene->m_materials, sizeof(Material) * materialNum);
    cudaMemcpy(hst_scene->m_materials, scene->m_materials.data(),
        sizeof(Material) * materialNum, cudaMemcpyHostToDevice);

    // Move Light Data
    int lightNum = scene->m_lights.size();
    cudaMalloc(&hst_scene->m_lights, sizeof(Light) * lightNum);
    cudaMemcpy(hst_scene->m_lights, scene->m_lights.data(),
        sizeof(Light) * lightNum, cudaMemcpyHostToDevice);

    // Move Primitive Data
    int primitiveNum = scene->m_primitives.size();
    cudaMalloc(&hst_scene->m_primitives, sizeof(Primitive) * primitiveNum);
    cudaMemcpy(hst_scene->m_primitives, scene->m_primitives.data(),
        sizeof(Primitive) * primitiveNum, cudaMemcpyHostToDevice);

    // Move cudaScene Data
    cudaMalloc(&dev_scene, sizeof(CUDAScene));
    cudaMemcpy(dev_scene, hst_scene, sizeof(CUDAScene), cudaMemcpyHostToDevice);

    // Move Camera Data
    hst_camera = &(renderer->m_camera);
    Film& film = hst_camera->m_film;
    cudaMalloc(&film.m_bitmap, film.m_resolution.x * film.m_resolution.y * sizeof(unsigned char));
    cudaMemset(film.m_bitmap, 0, film.m_resolution.x * film.m_resolution.y * sizeof(unsigned char));
    cudaMalloc(&dev_camera, sizeof(Camera));
    cudaMemcpy(dev_camera, hst_camera, sizeof(Camera), cudaMemcpyHostToDevice);

    // Move Integrator
    hst_integrator = &(renderer->m_integrator);
    cudaMalloc(&dev_integrator, sizeof(Integrator));
    cudaMemcpy(dev_integrator, hst_integrator, sizeof(Integrator), cudaMemcpyHostToDevice);
}

typedef struct
{
    float4 m[3];
} float3x4;

__constant__ float3x4 c_invViewMatrix;  // inverse view matrix

/*struct Ray
{
    float3 o;   // origin
    float3 d;   // direction
};

// intersect ray with a box
// http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm

__device__
int intersectBox(Ray r, float3 boxmin, float3 boxmax, float* tnear, float* tfar)
{
    // compute intersection of ray with all six bbox planes
    float3 invR = make_float3(1.0f) / r.d;
    float3 tbot = invR * (boxmin - r.o);
    float3 ttop = invR * (boxmax - r.o);

    // re-order intersections to find smallest and largest on each axis
    float3 tmin = fminf(ttop, tbot);
    float3 tmax = fmaxf(ttop, tbot);

    // find the largest tmin and the smallest tmax
    float largest_tmin = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
    float smallest_tmax = fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));

    *tnear = largest_tmin;
    *tfar = smallest_tmax;

    return smallest_tmax > largest_tmin;
}*/

// transform vector by matrix (no translation)
__device__
float3 mul(const float3x4& M, const float3& v)
{
    float3 r;
    r.x = dot(v, make_float3(M.m[0]));
    r.y = dot(v, make_float3(M.m[1]));
    r.z = dot(v, make_float3(M.m[2]));
    return r;
}

// transform vector by matrix with translation
__device__
float4 mul(const float3x4& M, const float4& v)
{
    float4 r;
    r.x = dot(v, M.m[0]);
    r.y = dot(v, M.m[1]);
    r.z = dot(v, M.m[2]);
    r.w = 1.0f;
    return r;
}

__device__ uint rgbaFloatToInt(float4 rgba)
{
    rgba.x = __saturatef(rgba.x);   // clamp to [0.0, 1.0]
    rgba.y = __saturatef(rgba.y);
    rgba.z = __saturatef(rgba.z);
    rgba.w = __saturatef(rgba.w);
    return (uint(rgba.w * 255) << 24) | (uint(rgba.z * 255) << 16) | (uint(rgba.y * 255) << 8) | uint(rgba.x * 255);
}


__global__ void
d_render(uint* d_output, uint imageW, uint imageH, CUDAScene* dev_scene)
{    
    const float3 boxMin = make_float3(-1.0f, -1.0f, -1.0f);
    const float3 boxMax = make_float3(1.0f, 1.0f, 1.0f);

    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;
    uint index = y * imageW + x;
    if (index == 0) {
        printf("%d\n", dev_scene->m_triangleMeshNum);
        printf("%d\n", dev_scene->m_triangleMeshes[0].m_triangleNum);
        for (int i = 0; i < dev_scene->m_triangleMeshes[0].m_triangleNum * 3; i++) {
            printf("%d ", dev_scene->m_triangleMeshes[0].m_indices[i]);
        }
        printf("\n");
        //for (int i = 0; i < a->f.size(); i++) {
            //printf("%d\n", a->f[i]);
        //}
        Vector3f v(1, 2, 3);
        printf("%f %f %f\n", v.x, v.y, v.z);
        Float len = v.Length();
        printf("%f\n", len);
    }

    if ((x >= imageW) || (y >= imageH)) return;

    //RandomSampler& sampler = samplers[index];
    //sampler.Init(x, y);

    //float u = ((x + sampler.Next()) / (float)imageW) * 2.0f - 1.0f;
    //float v = ((y + sampler.Next()) / (float)imageH) * 2.0f - 1.0f;

    float u = ((x) / (float)imageW) * 2.0f - 1.0f;
    float v = ((y) / (float)imageH) * 2.0f - 1.0f;

    // calculate eye ray in world space
    /*Ray eyeRay;
    eyeRay.o = make_float3(mul(c_invViewMatrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
    eyeRay.d = normalize(make_float3(u, v, -2.0f));
    eyeRay.d = mul(c_invViewMatrix, eyeRay.d);

    // find intersection with box
    float tnear, tfar;
    int hit = intersectBox(eyeRay, boxMin, boxMax, &tnear, &tfar);*/
    float tnear = 1;
    int hit = 0;

    if (!hit) return;

    if (tnear < 0.0f) tnear = 0.0f;     // clamp to near plane

    // march along ray from front to back, accumulating color
    float4 sum = make_float4(1.0f, 0.f, 0.f, 1.f);
   

    // write output color
    d_output[y * imageW + x] = rgbaFloatToInt(sum);

}

extern "C"
void freeCudaBuffers()
{
}


extern "C"
void render_kernel(dim3 gridSize, dim3 blockSize, uint * d_output, uint imageW, uint imageH)
{   
    d_render << <gridSize, blockSize >> > (d_output, imageW, imageH, dev_scene);
    cudaDeviceSynchronize();
    exit(0);
}

extern "C"
void copyInvViewMatrix(float* invViewMatrix, size_t sizeofMatrix)
{
    checkCudaErrors(cudaMemcpyToSymbol(c_invViewMatrix, invViewMatrix, sizeofMatrix));
}


#endif // #ifndef _VOLUMERENDER_KERNEL_CU_
