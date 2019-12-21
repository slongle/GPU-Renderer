#pragma once
#ifndef __MICROFACET_H
#define __MICROFACET_H

#include "renderer/core/fwd.h"
#include "renderer/core/geometry.h"
#include "renderer/core/sampling.h"

inline __device__ __host__
Vector3f Sample_wh(
    const Vector3f& wo, 
    unsigned int& seed) 
{
    Vector3f wh;
    bool flip = wo.z < 0;
    //wh = TrowbridgeReitzSample(flip ? -wo : wo, alphax, alphay, NextRandom(seed), NextRandom(seed));
    if (flip) wh = -wh;
    return wh;
}

#endif // !__MICROFACET_H