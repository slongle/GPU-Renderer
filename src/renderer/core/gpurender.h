#pragma once
#ifndef __GPURENDER_H
#define __GPURENDER_H

#include "renderer/core/renderer.h"

extern "C"
void gpu_render(std::shared_ptr<Renderer> renderer);

inline 
void GPURender(std::shared_ptr<Renderer> renderer){
    gpu_render(renderer);
}

#endif // !__GPURENDER_H
