#pragma once

#include "renderer/fwd.h"

struct Ray 
{
    float3 o;
    float3 d;
    float tMin;
    float tMax;
};

struct Hit 
{
    float t;
    int triangle_id;
    float u;
    float v;
};

struct Differential
{
    float3 p;
    float3 normal_s;
    float3 normal_g;
    float3 dpdu, dpdv;
};