#pragma once
#ifndef __SAMPLING_H
#define __SAMPLING_H

#include "renderer/core/fwd.h"

__device__ __host__
unsigned int RandomInit(
    unsigned int v0, 
    unsigned int v1, 
    unsigned int backoff = 16);

__device__ __host__
Float Next(
    unsigned int& seed);

#endif // __SAMPLING_H