#pragma once

#include <iostream>
#include <vector>
#include <map>

#include <vector_types.h>
#include <assert.h>

#include "utility/helper_string.h"
#include "utility/helper_math.h"
#include "utility/buffer.h"
#include "utility/memory_arena.h"

#include "renderer/imageio.h"

#include "ext/tinyformat/tinyformat.h"

#define HOST __host__
#define DEVICE __device__
#define HOST_DEVICE __host__ __device__

#define PI       3.14159265358979323846f
#define INV_PI   0.31830988618379067154f
#define INV_2_PI 0.15915494309189533577f

class Scene;

inline float degToRad(float deg)
{
    return deg / 180 * PI;
}

template<typename T>
inline T divideRoundInf(const T& x, const T& y)
{
    return (x + y - 1) / y;
}

