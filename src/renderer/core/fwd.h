#pragma once
#ifndef __FWD_H
#define __FWD_H

#include "utility/helper_logger.h"
#include "utility/helper_math.h"

#include "ext/tinyformat/tinyformat.h"
#include "ext/filesystem/resolver.h"


inline filesystem::resolver* getFileResolver() {
    static filesystem::resolver* resolver = new filesystem::resolver();
    return resolver;
}

typedef float Float;

#define Infinity ((float)(_HUGE_ENUF * _HUGE_ENUF))
#define Epsilon  1e-4f
#define Pi       3.14159265358979323846f
#define InvPi    0.31830988618379067154f
#define Inv2Pi   0.15915494309189533577f
#define Inv4Pi   0.07957747154594766788f
#define PiOver2  1.57079632679489661923f
#define PiOver4  0.78539816339744830961f
#define Sqrt2    1.41421356237309504880f

template<typename T> class Vector2;
template<typename T> class Vector3;
template<typename T> class Point2;
template<typename T> class Point3;
template<typename T> class Normal3;
template<typename T> class Bounds2;
template<typename T> class Bounds3;

typedef Vector2<Float> Vector2f;
typedef Vector2<int> Vector2i;
typedef Vector3<Float> Vector3f;
typedef Vector3<int> Vector3i;
typedef Point2<Float> Point2f;
typedef Point2<int> Point2i;
typedef Point3<Float> Point3f;
typedef Point3<int> Point3i;
typedef Normal3<Float> Normal3f;
typedef Bounds2<Float> Bounds2f;
typedef Bounds2<int> Bounds2i;
typedef Bounds3<Float> Bounds3f;
typedef Bounds3<int> Bounds3i;

class Ray;
class BSDF;
class Light;
class Primitive;
class Material;
class Medium;
class Spectrum;

inline __device__ __host__
Float Radians(Float ang) {
    return ang * Pi / 180.f;
}

template<typename T>
inline __device__ __host__
void Swap(T& a, T& b) {
    T c(a);
    a = b;
    b = c;
}

inline __device__ __host__
Float Clamp(Float a, Float l, Float r) {
    if (a > r) return r;
    else if (a < l) return l;
    else return a;
}

inline __device__ __host__
Float GammaCorrect(Float value) {
    if (value <= 0.0031308f)
        return 12.92f * value;
    return 1.055f * pow(value, (Float)(1.f / 2.4f)) - 0.055f;
}


#endif // !__FWD_H
