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
    __host__ __device__ Vector2(T x = 0, T y = 0) :x(x), y(y) {}

    // Vector2 Public Data
    T x, y;
};


template <typename T>
class Vector3 {
public:
    __host__ __device__ Vector3(T x = 0, T y = 0, T z = 0) :x(x), y(y), z(z) {}
    __host__ __device__ explicit Vector3<T>(Point3<T> p) : x(p.x), y(p.y), z(p.z) {}

    __host__ __device__ Vector3<T> operator  / (const Float f)const;

    __host__ __device__ Float Length() const;

    // Vector3 Public Data
    T x, y, z;
};

typedef Vector2<Float> Vector2f;
typedef Vector2<int> Vector2i;
typedef Vector3<Float> Vector3f;
typedef Vector3<int> Vector3i;

// Point Declarations
//template <typename T>
class Point2f {
public:
    __host__ __device__ Point2f(Float x = 0, Float y = 0) :x(x), y(y) {}

    // Point2 Public Data
    Float x, y;
};

class Point2i {
public:
    __host__ __device__ Point2i(int x = 0, int y = 0) :x(x), y(y) {}

    // Point2 Public Data
    int x, y;
};


template <typename T>
class Point3 {
public:
    __host__ __device__ Point3(T x = 0, T y = 0, T z = 0) :x(x), y(y), z(z) {}


    __host__ __device__ Vector3<T> operator - (const Point3<T>& p) const;
    __host__ __device__ Point3<T> operator / (T v) const;

    // Point3 Public Data
    T x, y, z;
};


//typedef Point2<Float> Point2f;
//typedef Point2<int> Point2i;
typedef Point3<Float> Point3f;
typedef Point3<int> Point3i;

// Normal Declarations
template <typename T>
class Normal3 {
public:
    __host__ __device__ Normal3(T x = 0, T y = 0, T z = 0) :x(x), y(y), z(z) {}

    // Normal3 Public Data
    T x, y, z;
};


typedef Normal3<Float> Normal3f;

// Bounds Declarations
template <typename T>
class Bounds2 {
public:
    

    // Bounds2 Public Data
    //Point2<T> pMin, pMax;
    Point2f pMin, pMax;
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
    __host__ __device__ Ray(Point3f o, Vector3f d, Float tMax = INFINITE) :o(o), d(d), tMax(tMax) {}

    // Ray Public Data
    Point3f o;
    Vector3f d;
    mutable Float tMax;
};




template<typename T>
inline Point3<T> Point3<T>::operator/(T v) const
{
    ASSERT(v != 0, "Divide zero");
    return Point3<T>(x / v, y / v, z / v);
}

template<typename T>
__host__ __device__
inline Vector3<T> Normalize(Vector3<T> v) {
    Float len = v.Length();
    return v / len;
}
template<typename T>
inline Vector3<T> Vector3<T>::operator/(const Float f) const
{
    ASSERT(f != 0, "Divide zero");
    return Vector3<T>(x / f, y / f, z / f);
}

template<typename T>
inline Float Vector3<T>::Length() const
{
    return std::sqrt(x * x + y * y + z * z);
}

template<typename T>
inline Vector3<T> Point3<T>::operator-(const Point3<T>& p) const
{
    return Vector3<T>(x - p.x, y - p.y, z - p.z);
}


template<typename T>
__host__ __device__
inline Vector3<T> Cross(const Vector3<T>& v1, const Vector3<T>& v2) {
    return Vector3<T>(v1.y * v2.z - v1.z * v2.y,
        v1.z * v2.x - v1.x * v2.z,
        v1.x * v2.y - v1.y * v2.x);
}

template<typename T>
__host__ __device__
inline T Dot(const Vector3<T>& v1, const Vector3<T>& v2) {
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

#endif // !__VECTOR_H
