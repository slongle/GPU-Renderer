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
#include "renderer/core/geometry.h"

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

unsigned int frame = 0;

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
    cudaMalloc(&film.m_bitmap, film.m_resolution.x * film.m_resolution.y * 3 * sizeof(Float));
    cudaMemset(film.m_bitmap, 0, film.m_resolution.x * film.m_resolution.y * 3 * sizeof(Float));
    cudaMalloc(&film.m_sampleNum, film.m_resolution.x * film.m_resolution.y * sizeof(unsigned int));
    cudaMemset(film.m_sampleNum, 0, film.m_resolution.x * film.m_resolution.y * sizeof(unsigned int));
    cudaMalloc(&dev_camera, sizeof(Camera));
    checkCudaErrors(cudaMemcpy(dev_camera, hst_camera, sizeof(Camera), cudaMemcpyHostToDevice));

    // Move Integrator
    hst_integrator = &(renderer->m_integrator);
    cudaMalloc(&dev_integrator, sizeof(Integrator));
    cudaMemcpy(dev_integrator, hst_integrator, sizeof(Integrator), cudaMemcpyHostToDevice);

    hst_renderer = new CUDARenderer(dev_integrator, dev_camera, dev_scene);
    cudaMalloc(&dev_renderer, sizeof(CUDARenderer));
    checkCudaErrors(cudaMemcpy(dev_renderer, hst_renderer, sizeof(CUDARenderer), cudaMemcpyHostToDevice));


}

typedef struct
{
    float4 m[3];
} float3x4;

__constant__ float3x4 c_invViewMatrix;  // inverse view matrix

inline __device__
Float PowerHeuristic(int nf, Float fPdf, int ng, Float gPdf) {
    Float f = nf * fPdf, g = ng * gPdf;
    return (f * f) / (f * f + g * g);
}

inline __device__
Spectrum NextEventEstimate(const CUDAScene& scene, const Interaction& inter, unsigned int& seed, Point3f& pLight) 
{    
    const Primitive& primitive = scene.m_primitives[inter.m_primitiveID];
    const Material& material = scene.m_materials[primitive.m_materialID];

    Spectrum est;
    
    // Sample one of lights
    int lightID = min(scene.m_lightNum - 1, int(NextRandom(seed) * scene.m_lightNum));
    Float lightChoosePdf = Float(1) / scene.m_lightNum;
    const Light& light = scene.m_lights[lightID];

    // Light Sampling
    {
        // Light Sample Li
        const Triangle& triangle = scene.m_triangles[light.m_shapeID];
        Float lightSamplePdf;
        Interaction lightSample = triangle.Sample(&lightSamplePdf, seed);
        pLight = lightSample.m_p;
        lightSamplePdf *= (lightSample.m_p - inter.m_p).SqrLength() / 
            AbsDot(-Normalize(lightSample.m_p - inter.m_p), lightSample.m_shadingN);

        // Visibility test
        Point3f origin = inter.m_p + Normalize(lightSample.m_p - inter.m_p) * Epsilon;
        Point3f target = lightSample.m_p + Normalize(origin - lightSample.m_p) * Epsilon;
        Vector3f d = target - origin;
        Ray testRay(origin, Normalize(d), d.Length() - Epsilon);
        bool hit = scene.Intersect(testRay);

        if (!hit) {
            Vector3f d = Normalize(lightSample.m_p - inter.m_p);
            // Get Le
            Spectrum Le(0.);
            if (Dot(-d, lightSample.m_shadingN) > 0) {
                Le = light.m_L;
            }

            // BSDF Sample
            Normal3f n = inter.m_shadingN;
            Float bsdfPdf;
            Spectrum cosBSDF;
            cosBSDF = material.F(n, inter.m_wo, d, &bsdfPdf);

            // Contribution
            if (light.isDelta()) {
                est += Le * cosBSDF / lightSamplePdf;
            }
            else {
                Float weight = PowerHeuristic(1, lightSamplePdf, 1, bsdfPdf);                
                est += Le * cosBSDF * weight / lightSamplePdf;
            }
        }
    }

    // BSDF Sampling
    if (!light.isDelta()) {

        // BSDF Sample
        Normal3f n = inter.m_shadingN;
        Float bsdfPdf;
        Spectrum cosBSDF;
        Vector3f wi;
        cosBSDF = material.Sample(n, inter.m_wo, &wi, &bsdfPdf, seed);

        // Light Sample
        const Triangle& triangle = scene.m_triangles[light.m_shapeID];

        Point3f origin = inter.m_p + wi * Epsilon;
        Ray testRay(origin, wi);
        Interaction lightInter;
        bool hit = scene.IntersectP(testRay, &lightInter);

        if (hit && scene.m_primitives[lightInter.m_primitiveID].m_lightID != -1)
        {
            Float lightSamplePdf;
            lightSamplePdf = (lightInter.m_p - inter.m_p).SqrLength() /
                (AbsDot(-wi, lightInter.m_shadingN) * triangle.Area());
            pLight = lightInter.m_p;

            // Get Le            
            Spectrum Le(0.);
            if (Dot(-wi, lightInter.m_shadingN) > 0) {
                Le = scene.m_lights[scene.m_primitives[lightInter.m_primitiveID].m_lightID].m_L;
            }

            Float weight = PowerHeuristic(1, bsdfPdf, 1, lightSamplePdf);
            est += Le * cosBSDF * weight / bsdfPdf;            
        }
    }

    return est / lightChoosePdf;
}

inline __device__
Spectrum SampleMaterial(const CUDAScene& scene, Interaction& inter, unsigned int& seed) {
    Spectrum cosBSDF;
    const Primitive& primitive = scene.m_primitives[inter.m_primitiveID];
    const Material& material = scene.m_materials[primitive.m_materialID];
    
    Normal3f n = inter.m_shadingN;

    Float bsdfPdf;
    cosBSDF = material.Sample(n, inter.m_wo, &inter.m_wi, &bsdfPdf, seed);

    return cosBSDF / bsdfPdf;
}

__global__ void
d_render(uint* d_output, uint imageW, uint imageH, unsigned int frame, CUDARenderer* renderer)
{
    const float3 boxMin = make_float3(-1.0f, -1.0f, -1.0f);
    const float3 boxMax = make_float3(1.0f, 1.0f, 1.0f);

    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;

    Integrator* integrator = renderer->m_integrator;
    Camera* camera = renderer->m_camera;
    CUDAScene* scene = renderer->m_scene;

    uint index = y * imageW + x;
    if ((x >= imageW) || (y >= imageH)) return;

    Spectrum L(0);    
    uint seed = InitRandom(index, frame);
    Spectrum throughput(1);
    Ray ray = camera->GenerateRay(Point2f(x + NextRandom(seed), y + NextRandom(seed)));
    int bounce;
    for (bounce = 0; bounce < integrator->m_maxDepth; bounce++) {

        // find intersection with scene
        Interaction interaction;
        bool hit = scene->IntersectP(ray, &interaction);

        if (!hit) {
            break;
        }

        const Primitive& primitive = scene->m_primitives[interaction.m_primitiveID];
        if (bounce == 0 && primitive.m_lightID != -1) {
            int lightID = primitive.m_lightID;
            const Light& light = scene->m_lights[lightID];
            if (Dot(interaction.m_shadingN, interaction.m_wo) > 0) {
                L += throughput * light.m_L;
            }
        }

        // render normal
        //L = Spectrum(interaction.m_geometryN);
        //break;

        if (throughput.isBlack()) {
            break;
        }

        // direct light
        Point3f pLight;
        L += throughput * NextEventEstimate(*scene, interaction, seed, pLight);

        // calculate BSDF
        throughput *= SampleMaterial(*scene, interaction, seed);

        // indirect light                    
        if (throughput.Max() < 1 && bounce > 3) {
            Float q = max((Float).05, 1 - throughput.Max());
            if (NextRandom(seed) < q) break;
            throughput /= 1 - q;
        }

        ray.o = interaction.m_p + interaction.m_wi * Epsilon;
        ray.d = interaction.m_wi;
        ray.tMax = Infinity;
    }
    camera->m_film.AddSample(x, y, L);
    L = camera->m_film.GetPixelSpectrum(index);

    // write output color
    if (d_output) {
        SpectrumToUnsignedChar(L, (unsigned char*)&d_output[(imageH - y - 1) * imageW + x], 4);
    }
    else {
        //printf("0");
    }

}

extern "C"
void freeCudaBuffers()
{
}


extern "C"
void render_kernel(dim3 gridSize, dim3 blockSize, uint * d_output, uint imageW, uint imageH)
{    
    d_render << <gridSize, blockSize >> > (d_output, imageW, imageH, frame, dev_renderer);
    //checkCudaErrors(cudaDeviceSynchronize());
    frame++;
}

extern "C"
void copyInvViewMatrix(float* invViewMatrix, size_t sizeofMatrix)
{
    //checkCudaErrors(cudaMemcpyToSymbol(c_invViewMatrix, invViewMatrix, sizeofMatrix));
}

extern "C"
void gpu_render(std::shared_ptr<Renderer> renderer) {
    auto iDivUp = [](unsigned int a, unsigned int b)->unsigned int {
        return (a + b - 1) / b;
    };

    cudaInit(renderer);

    int width = renderer->m_camera.m_film.m_resolution.x;
    int height = renderer->m_camera.m_film.m_resolution.y;
    Camera* camera = &renderer->m_camera;
    Film film = camera->m_film;
    
    dim3 blockSize{ 16, 16 };
    dim3 gridSize{ iDivUp(width, blockSize.x), iDivUp(height, blockSize.y) };
    for (unsigned int i = 0; i < renderer->m_integrator.m_nSample; i++) {
        d_render << <gridSize, blockSize >> > (NULL, width, height, i, dev_renderer);
    }
    checkCudaErrors(cudaDeviceSynchronize());

    film.m_bitmap = new Float[film.m_resolution.x * film.m_resolution.y * 3];
    film.m_sampleNum = new unsigned int[film.m_resolution.x * film.m_resolution.y];
    memset(film.m_bitmap, 0, sizeof(Float) * film.m_resolution.x * film.m_resolution.y * 3);
    memset(film.m_sampleNum, 0, sizeof(unsigned int) * film.m_resolution.x * film.m_resolution.y);    
    checkCudaErrors(cudaMemcpy(film.m_bitmap, hst_camera->m_film.m_bitmap, sizeof(Float) * film.m_resolution.x * film.m_resolution.y * 3, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(film.m_sampleNum, hst_camera->m_film.m_sampleNum, sizeof(unsigned int) * film.m_resolution.x * film.m_resolution.y, cudaMemcpyDeviceToHost));
    //cudaDeviceSynchronize();
    film.Output();     
}

#endif // #ifndef _VOLUMERENDER_KERNEL_CU_
