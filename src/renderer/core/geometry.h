#pragma once
#ifndef __GEOMETRY_H
#define __GEOMETRY_H

#include "renderer/core/fwd.h"

template <typename T>
inline bool isNaN(const T x) {
    return std::isnan(x);
}
template <>
inline bool isNaN(const int x) {
    return false;
}

// Vector Declarations
template <typename T>
class Vector2 {
public:
    Vector2(T x = 0, T y = 0) :x(x), y(y) {}

    // Vector2 Public Data
    T x, y;
};


template <typename T>
class Vector3 {
public:
    Vector3(T x = 0, T y = 0, T z = 0) :x(x), y(y), z(z) {}

    // Vector3 Public Data
    T x, y, z;
};

typedef Vector2<Float> Vector2f;
typedef Vector2<int> Vector2i;
typedef Vector3<Float> Vector3f;
typedef Vector3<int> Vector3i;

// Point Declarations
template <typename T>
class Point2 {
public:
    Point2(T x = 0, T y = 0) :x(x), y(y) {}

    // Point2 Public Data
    T x, y;
};


template <typename T>
class Point3 {
public:
    Point3(T x = 0, T y = 0, T z = 0) :x(x), y(y), z(z) {}

    // Point3 Public Data
    T x, y, z;
};


typedef Point2<Float> Point2f;
typedef Point2<int> Point2i;
typedef Point3<Float> Point3f;
typedef Point3<int> Point3i;

// Normal Declarations
template <typename T>
class Normal3 {
public:
    Normal3(T x = 0, T y = 0, T z = 0) :x(x), y(y), z(z) {}

    // Normal3 Public Data
    T x, y, z;
};


typedef Normal3<Float> Normal3f;

// Bounds Declarations
template <typename T>
class Bounds2 {
public:
    

    // Bounds2 Public Data
    Point2<T> pMin, pMax;
};

template <typename T>
class Bounds3 {
public:
    
    // Bounds3 Public Data
    Point3<T> pMin, pMax;
};

typedef Bounds2<Float> Bounds2f;
typedef Bounds2<int> Bounds2i;
typedef Bounds3<Float> Bounds3f;
typedef Bounds3<int> Bounds3i;

// Ray Declarations
class Ray {
public:

    // Ray Public Data
    Point3f o;
    Vector3f d;
    mutable Float tMax;
    const Medium* medium;
};


#endif // !__VECTOR_H
