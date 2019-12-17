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
#include "renderer/core/sampling.h"
#include "renderer/core/camera.h"

#include "renderer/kernel/cudascene.h"
#include "renderer/kernel/cudarenderer.h"

CUDAScene* hst_scene;
CUDAScene* dev_scene;
Camera* hst_camera;
Camera* dev_camera;
Integrator* hst_integrator;
Integrator* dev_integrator;
CUDARenderer* hst_renderer;
CUDARenderer* dev_renderer;

//int frame = 0;

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
    for (int i = 0; i < triangleNum; i++) {
        int meshID = scene->m_triangles[i].m_triangleMeshID;
        scene->m_triangles[i].m_triangleMeshPtr = triangleMeshGPUPtr + meshID;
    }
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

    hst_renderer = new CUDARenderer(dev_integrator, dev_camera, dev_scene);
    cudaMalloc(&dev_renderer, sizeof(CUDARenderer));
    cudaMemcpy(dev_renderer, hst_renderer, sizeof(CUDARenderer), cudaMemcpyHostToDevice);
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

__global__ void
d_render(uint* d_output, uint imageW, uint imageH, int frame, CUDARenderer* renderer)
{
    const float3 boxMin = make_float3(-1.0f, -1.0f, -1.0f);
    const float3 boxMax = make_float3(1.0f, 1.0f, 1.0f);

    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;

    Integrator* integrator = renderer->m_integrator;
    Camera* camera = renderer->m_camera;
    CUDAScene* scene = renderer->m_scene;

    uint index = y * imageW + x;
    /*
    if (index == 0) {
        printf("%d\n", scene->m_triangleMeshNum);
        printf("%d\n", scene->m_triangleMeshes[0].m_triangleNum);
        for (int i = 0; i < scene->m_triangleMeshes[0].m_triangleNum * 3; i++) {
            printf("%d ", scene->m_triangleMeshes[0].m_indices[i]);
        }
        printf("\n");
        //for (int i = 0; i < a->f.size(); i++) {
            //printf("%d\n", a->f[i]);
        //}
        Vector3f v(1, 2, 3);
        printf("%f %f %f\n", v.x, v.y, v.z);
        Float len = v.Length();
        printf("%f\n", len);
    }*/

    if ((x >= imageW) || (y >= imageH)) return;

    uint seed = InitRandom(index, frame);
    Spectrum L(0);
    Spectrum throughput(1);

    Ray ray = camera->GenerateRay(Point2f(x + NextRandom(seed), y + NextRandom(seed)));
    for (int i = 0; i < integrator->m_maxDepth; i++) {

        // find intersection with scene
        Interaction interaction;
        bool hit = scene->IntersectP(ray, &interaction);

        if (!hit){
            break;
        }

        // direct light
        L += throughput * scene->SampleLight(interaction, seed);
        
        // calculate BSDF
        throughput *= scene->Shading(interaction, seed);
        if (throughput.isBlack()) {
            break;
        }

        // indirect light
        if (i > 3) {
            Float q = min(Float(0.95), throughput.Max());
            if (NextRandom(seed) >= q) {
                break;
            }
            throughput /= q;
        }

        ray.o = interaction.m_p + interaction.m_geometryN * EPSILON;
        ray.d = -interaction.m_wi;
        ray.tMax = INFINITY;
    }

    // write output color
    SpectrumToUnsignedChar(L, (unsigned char*)&d_output[(imageH - y - 1) * imageW + x], 4);

}

extern "C"
void freeCudaBuffers()
{
}


extern "C"
void render_kernel(dim3 gridSize, dim3 blockSize, uint * d_output, uint imageW, uint imageH)
{
    d_render << <gridSize, blockSize >> > (d_output, imageW, imageH, 0, dev_renderer);
    //cudaDeviceSynchronize();
    //frame++;
    //exit(0);
}

extern "C"
void copyInvViewMatrix(float* invViewMatrix, size_t sizeofMatrix)
{
    checkCudaErrors(cudaMemcpyToSymbol(c_invViewMatrix, invViewMatrix, sizeofMatrix));
}


#endif // #ifndef _VOLUMERENDER_KERNEL_CU_
