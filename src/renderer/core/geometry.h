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
    __host__ __device__ explicit Vector3<T>(Normal3<T> n) : x(n.x), y(n.y), z(n.z) {}

    __host__ __device__ Vector3<T> operator - ()const;
    __host__ __device__ Vector3<T> operator + (const Vector3<T>& v)const;
    __host__ __device__ Vector3<T> operator + (const Normal3<T>& v)const;
    __host__ __device__ Vector3<T> operator * (const Float f)const;
    __host__ __device__ Vector3<T> operator / (const Float f)const;

    __host__ __device__ Float Length() const;
    __host__ __device__ Float SqrLength() const;

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
    __host__ __device__ Point2(T x = 0, T y = 0) :x(x), y(y) {}
    __host__ __device__ Point2(const Point3<T>& p) : x(p.x), y(p.y) {}

    __host__ __device__ Point2<T> operator + (const Point2<T>& p) const;
    __host__ __device__ Point2<T> operator * (const Float g) const;

    // Point2 Public Data
    T x, y;
};


template <typename T>
class Point3 {
public:
    __host__ __device__ Point3(T x = 0, T y = 0, T z = 0) :x(x), y(y), z(z) {}


    __host__ __device__ Point3<T> operator + (const Vector3<T>& v) const;
    __host__ __device__ Point3<T> operator + (const Point3<T>& v) const;
    __host__ __device__ Vector3<T> operator - (const Point3<T>& p) const;
    __host__ __device__ Point3<T> operator * (T v) const;
    __host__ __device__ Point3<T> operator / (T v) const;

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
    __host__ __device__ Normal3(T x = 0, T y = 0, T z = 0) :x(x), y(y), z(z) {}
    __host__ __device__ explicit Normal3(Vector3<T> v) : x(v.x), y(v.y), z(v.z) {}

    __host__ __device__ Normal3<T> operator - () const;
    __host__ __device__ Normal3<T> operator + (const Normal3<T>& n) const;
    __host__ __device__ Normal3<T> operator * (Float f) const;
    __host__ __device__ Normal3<T> operator / (Float f) const;

    __host__ __device__ Float Length() const;

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
    __host__ __device__ Ray(Point3f o, Vector3f d, Float tMax = Infinity) :o(o), d(d), tMax(tMax) {}
    
    __host__ __device__ Point3f operator() (Float t) const;

    // Ray Public Data
    Point3f o;
    Vector3f d;
    mutable Float tMax;
};

template<typename T>
inline __host__ __device__
Point3<T> Point3<T>::operator/(T v) const
{
    ASSERT(v != 0, "Divide zero");
    return Point3<T>(x / v, y / v, z / v);
}

template<typename T>
inline __host__ __device__
Vector3<T> Normalize(Vector3<T> v) {
    Float len = v.Length();
    return v / len;
}

template<typename T>
inline __host__ __device__
Normal3<T> Normalize(Normal3<T> v) {
    Float len = v.Length();
    return v / len;
}


template<typename T>
inline __host__ __device__ 
Vector3<T> Vector3<T>::operator-() const
{
    return Vector3<T>(-x, -y, -z);
}

template<typename T>
inline __host__ __device__ 
Vector3<T> Vector3<T>::operator+(const Vector3<T>& v) const
{
    return Vector3<T>(x + v.x, y + v.y, z + v.z);
}

template<typename T>
inline __host__ __device__ 
Vector3<T> Vector3<T>::operator+(const Normal3<T>& v) const
{
    return Vector3<T>(x + v.x, y + v.y, z + v.z);
}

template<typename T>
inline __host__ __device__ 
Vector3<T> Vector3<T>::operator*(const Float f) const
{
    return Vector3<T>(x * f, y * f, z * f);
}

template<typename T>
inline __host__ __device__
Vector3<T> Vector3<T>::operator/(const Float f) const
{
    ASSERT(f != 0, "Divide zero");
    Float invF = 1 / f;
    return Vector3<T>(x * invF, y * invF, z * invF);
}

template<typename T>
inline __host__ __device__
Float Vector3<T>::Length() const
{
    return sqrt(x * x + y * y + z * z);
}

template<typename T>
inline __host__ __device__ 
Float Vector3<T>::SqrLength() const
{
    return x * x + y * y + z * z;
}

template<typename T>
inline __host__ __device__ 
Point3<T> Point3<T>::operator+(const Vector3<T>& v) const
{
    return Point3<T>(x + v.x, y + v.y, z + v.z);
}

template<typename T>
inline __host__ __device__ 
Point3<T> Point3<T>::operator+(const Point3<T>& v) const
{
    return Point3<T>(x + v.x, y + v.y, z + v.z);
}

template<typename T>
inline __host__ __device__
Vector3<T> Point3<T>::operator-(const Point3<T>& p) const
{
    return Vector3<T>(x - p.x, y - p.y, z - p.z);
}

template<typename T>
inline __host__ __device__ 
Point3<T> Point3<T>::operator*(T v) const
{
    return Point3<T>(x * v, y * v, z * v);
}

inline __host__ __device__
Point3f Ray::operator() (Float t) const {
    return o + d * t;
}

template<typename T>
inline __host__ __device__ 
Normal3<T> Normal3<T>::operator-() const
{
    return Normal3<T>(-x, -y, -z);
}

template<typename T>
inline __host__ __device__ 
Normal3<T> Normal3<T>::operator+(const Normal3<T>& n) const
{
    return Normal3<T>(x + n.x, y + n.y, z + n.z);
}

template<typename T>
inline __host__ __device__ 
Normal3<T> Normal3<T>::operator*(Float f) const
{
    return Normal3<T>(x * f, y * f, z * f);
}

template<typename T>
inline __host__ __device__ Normal3<T> Normal3<T>::operator/(Float f) const
{
    Float invF = 1. / f;
    return Normal3<T>(x * invF, y * invF, z * invF);
}

template<typename T>
inline __host__ __device__ Float Normal3<T>::Length() const
{
    return sqrt(x * x + y * y + z * z);
}

template<typename T>
inline __host__ __device__ 
Point2<T> Point2<T>::operator+(const Point2<T>& p) const
{
    return Point2<T>(x + p.x, y + p.y);
}

template<typename T>
inline __host__ __device__ 
Point2<T> Point2<T>::operator*(const Float g) const
{
    return Point2<T>(x * g, y * g);
}

template<typename T>
__host__ __device__
Point3<T> operator + (const Point3<T>& p, const Normal3<T>& n) {
    return Point3<T>(p.x + n.x, p.y + n.y, p.z + n.z);
}

template<typename T>
inline __host__ __device__
Vector3<T> Cross(const Vector3<T>& v1, const Vector3<T>& v2) {
    return Vector3<T>(v1.y * v2.z - v1.z * v2.y,
        v1.z * v2.x - v1.x * v2.z,
        v1.x * v2.y - v1.y * v2.x);
}

template<typename T>
inline __host__ __device__
Vector3<T> Cross(const Vector3<T>& v1, const Normal3<T>& v2) {
    return Vector3<T>(v1.y * v2.z - v1.z * v2.y,
        v1.z * v2.x - v1.x * v2.z,
        v1.x * v2.y - v1.y * v2.x);
}

template<typename T>
inline __host__ __device__
Vector3<T> Cross(const Normal3<T>& v1, const Vector3<T>& v2) {
    return Vector3<T>(v1.y * v2.z - v1.z * v2.y,
        v1.z * v2.x - v1.x * v2.z,
        v1.x * v2.y - v1.y * v2.x);
}

template<typename T>
inline __host__ __device__
T Dot(const Vector3<T>& v1, const Vector3<T>& v2) {
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

template<typename T>
inline __host__ __device__
T Dot(const Normal3<T>& n, const Vector3<T>& v) {
    return n.x * v.x + n.y * v.y + n.z * v.z;
}

template<typename T>
inline __host__ __device__
T Dot(const Vector3<T>& v, const Normal3<T>& n) {
    return n.x * v.x + n.y * v.y + n.z * v.z;
}

template<typename T>
inline __host__ __device__
T AbsDot(const Vector3<T>& v1, const Vector3<T>& v2) {
    return abs(v1.x * v2.x + v1.y * v2.y + v1.z * v2.z);
}

template<typename T>
inline __host__ __device__
T AbsDot(const Normal3<T>& n, const Vector3<T>& v) {
    return abs(n.x * v.x + n.y * v.y + n.z * v.z);
}

template<typename T>
inline __host__ __device__
T AbsDot(const Vector3<T>& v, const Normal3<T>& n) {
    return abs(n.x * v.x + n.y * v.y + n.z * v.z);
}


#endif // !__VECTOR_H