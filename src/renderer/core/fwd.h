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


#define INFINITY   ((float)(_HUGE_ENUF * _HUGE_ENUF))
#define EPSILON 1e-2f
#define PI 3.14159265358979323846f

typedef float Float;

template<typename T> class Vector2;
template<typename T> class Vector3;
//template<typename T> class Point2;
template<typename T> class Point3;
template<typename T> class Normal3;
template<typename T> class Bounds2;
template<typename T> class Bounds3;

typedef Vector2<Float> Vector2f;
typedef Vector2<int> Vector2i;
typedef Vector3<Float> Vector3f;
typedef Vector3<int> Vector3i;
//typedef Point2<Float> Point2f;
//typedef Point2<int> Point2i;
typedef Point3<Float> Point3f;
typedef Point3<int> Point3i;
typedef Normal3<Float> Normal3f;
typedef Bounds2<Float> Bounds2f;
typedef Bounds2<int> Bounds2i;
typedef Bounds3<Float> Bounds3f;
typedef Bounds3<int> Bounds3i;

class Light;
class Shape;
class Primitive;
class Material;
class Medium;

inline __device__ __host__
Float Radians(Float ang) {
    return ang * PI / 180;
}


#endif // !__FWD_H
