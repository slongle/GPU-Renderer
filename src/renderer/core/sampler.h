#pragma once
#ifndef __SAMPLER_H
#define __SAMPLER_H

#include "renderer/core/fwd.h"
#include "renderer/core/parameterset.h"


class Sampler {
public:
    __device__ __host__ Sampler(int sampleNum = 0) :m_sampleNum(sampleNum) {}

    __device__ __host__ virtual void Init(
        unsigned int v0, 
        unsigned int v1, 
        unsigned int backoff = 16) = 0;

    __device__ __host__ virtual Float Next() = 0;

    int m_sampleNum;
};

#endif // !__SAMPLER_H
