#pragma once

#include "renderer/fwd.h"

typedef float3 Spectrum;

inline HOST_DEVICE
bool isBlack(const Spectrum& s) 
{
    return s.x == 0 && s.y == 0 && s.z == 0;
}

inline HOST_DEVICE
float GammaCorrect(float value) 
{
    if (value <= 0.0031308f)
        return 12.92f * value;
    /*if (isnan(powf(value, (1.f / 2.4f)))) {
        printf("%f\n", value);

    }*/
    return 1.055f * powf(value, (1.f / 2.4f)) - 0.055f;
}
