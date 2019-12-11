#pragma once
#ifndef __RANDOM_H
#define __RANDOM_H

#include "renderer/core/sampler.h"

class RandomSampler{// :public Sampler{
public:
    RandomSampler(int sampleNum = 0) :m_samplerNum(sampleNum) {}

    __device__ __host__ void Init(
        unsigned int v0,
        unsigned int v1,
        unsigned int backoff = 16);// override;
    
    __device__ __host__ Float Next();// override;


    unsigned int m_seed;
    int m_samplerNum;
};

#endif // !__RANDOM_H
