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

    T operator [] (int idx) const;
    Vector3<T> operator - ()const;
    Vector3<T> operator + (const Vector3<T>& v)const;
    Vector3<T> operator + (const Normal3<T>& v)const;
    Vector3<T> operator * (const Float f)const;
    Vector3<T> operator / (const Float f)const;

    Float Length() const;
    Float SqrLength() const;

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

    Point2<T> operator + (const Point2<T>& p) const;
    Point2<T> operator * (const Float g) const;

    // Point2 Public Data
    T x, y;
};


template <typename T>
class Point3 {
public:
    __host__ __device__ Point3(T x = 0, T y = 0, T z = 0) :x(x), y(y), z(z) {}

    T operator [] (int idx) const;
    Point3<T> operator + (const Vector3<T>& v) const;
    Point3<T> operator + (const Point3<T>& v) const;
    Vector3<T> operator - (const Point3<T>& p) const;
    Point3<T> operator * (T v) const;
    Point3<T> operator / (T v) const;

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

    Normal3<T> operator - () const;
    Normal3<T> operator + (const Normal3<T>& n) const;
    Normal3<T> operator * (Float f) const;
    Normal3<T> operator / (Float f) const;

    Float Length() const;

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
    __host__ __device__ Bounds3() :pMin(Point3<T>(99999, 99999, 99999)),pMax(Point3<T>(-99999, -99999,-99999)){}
    __host__ __device__ Bounds3(const Point3<T>& p) :pMin(p), pMax(p) {}
    __host__ __device__ Bounds3(const Point3<T>& p1, const Point3<T>& p2)
        : pMin(min(p1.x, p2.x), min(p1.y, p2.y), min(p1.z, p2.z)),
          pMax(max(p1.x, p2.x), max(p1.y, p2.y), max(p1.z, p2.z))
    {
    }
    
    Point3<T> operator [] (int idx) const;
    Vector3<T> Offset(const Point3<T>& p) const;
    Vector3<T> Diagonal() const;
    Point3<T> Centroid() const;
    int MaximumExtent() const;
    T Area() const;



    bool Intersect(const Ray& ray, Float* hitt0 = nullptr, Float* hitt1 = nullptr) const;
    bool Intersect(const Ray& ray, const Vector3f& invDir, const int dirIsNeg[3]) const;
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
    
    Point3f operator() (Float t) const;

    // Ray Public Data
    Point3f o;
    Vector3f d;
    mutable Float tMax;
};

template<typename T>
inline __device__ __host__
Point3<T> Min(const Point3<T>& p1, const Point3<T>& p2) {
    return Point3<T>(min(p1.x, p2.x),
        min(p1.y, p2.y),
        min(p1.z, p2.z));
}

template<typename T>
inline __device__ __host__
Point3<T> Max(const Point3<T>& p1, const Point3<T>& p2) {
    return Point3<T>(max(p1.x, p2.x),
        max(p1.y, p2.y),
        max(p1.z, p2.z));
}

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
T Vector3<T>::operator[](int idx) const
{
    if (idx == 0) return x;
    else if (idx == 1) return y;
    else return z;
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
T Point3<T>::operator[](int idx) const
{
    if (idx == 0) return x;
    else if (idx == 1) return y;
    else return z;
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
    Float invF = 1.f / f;
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

template<typename T>
inline __host__ __device__
Normal3<T> Faceforward(const Normal3<T>& n, const Vector3<T>& v) {
    if (Dot(v, n) < 0.f) {
        return -n;
    }
    else {
        return n;
    }
}

template<typename T>
inline __device__ __host__
Point3<T> Bounds3<T>::operator[](int idx) const
{
    if (idx == 0) return pMin;
    else return pMax;
}

template<typename T>
inline __device__ __host__
Vector3<T> Bounds3<T>::Offset(const Point3<T>& p) const
{
    Vector3<T> ret = p - pMin;
    if (pMax.x > pMin.x) ret.x /= pMax.x - pMin.x;
    if (pMax.y > pMin.y) ret.y /= pMax.y - pMin.y;
    if (pMax.z > pMin.z) ret.z /= pMax.z - pMin.z;
    return ret;
}

template<typename T>
inline __device__ __host__
Vector3<T> Bounds3<T>::Diagonal() const
{
    return pMax - pMin;
}

template<typename T>
inline __device__ __host__
int Bounds3<T>::MaximumExtent() const
{
    Vector3<T> diag = Diagonal();
    if (diag.x > diag.y&& diag.x > diag.z)
        return 0;
    else if (diag.y > diag.z)
        return 1;
    else return 2;
}

template<typename T>
inline __device__ __host__
T Bounds3<T>::Area() const
{
    Vector3<T> det = Diagonal();
    return 2 * (det.x * det.y + det.y * det.z + det.x * det.z);
}

template<typename T>
inline __device__ __host__
Point3<T> Bounds3<T>::Centroid() const
{
    return (pMin + pMax) * 0.5f;
}

template<typename T>
inline __device__ __host__
Bounds3<T> Union(const Bounds3<T>& b1, const Point3<T>& b2)
{
    Bounds3<T> bounds;
    bounds.pMin = Min(b1.pMin, b2);
    bounds.pMax = Max(b1.pMax, b2);
    return bounds;
}

template<typename T>
inline __device__ __host__
Bounds3<T> Union(const Bounds3<T>& b1, const Bounds3<T>& b2) 
{
    Bounds3<T> bounds;
    bounds.pMin = Min(b1.pMin, b2.pMin);
    bounds.pMax = Max(b1.pMax, b2.pMax);
    return bounds;
}

template <typename T>
inline __device__ __host__
bool Bounds3<T>::Intersect(
    const Ray& ray, 
    Float* hitt0,
    Float* hitt1) const 
{
    Float t0 = 0, t1 = ray.tMax;
    for (int i = 0; i < 3; ++i) {
        // Update interval for _i_th bounding box slab
        Float invRayDir = 1 / ray.d[i];
        Float tNear = (pMin[i] - ray.o[i]) * invRayDir;
        Float tFar = (pMax[i] - ray.o[i]) * invRayDir;

        // Update parametric interval from slab intersection $t$ values
        if (tNear > tFar) std::swap(tNear, tFar);

        // ray--bounds intersection
        t0 = tNear > t0 ? tNear : t0;
        t1 = tFar < t1 ? tFar : t1;
        if (t0 > t1) return false;
    }
    if (hitt0) *hitt0 = t0;
    if (hitt1) *hitt1 = t1;
    return true;
}

template <typename T>
inline __device__ __host__
bool Bounds3<T>::Intersect(
    const Ray& ray, 
    const Vector3f& invDir,
    const int dirIsNeg[3]) const 
{
    const Bounds3f& bounds = *this;
    // Check for ray intersection against $x$ and $y$ slabs
    Float tMin = (bounds[dirIsNeg[0]].x - ray.o.x) * invDir.x;
    Float tMax = (bounds[1 - dirIsNeg[0]].x - ray.o.x) * invDir.x;
    Float tyMin = (bounds[dirIsNeg[1]].y - ray.o.y) * invDir.y;
    Float tyMax = (bounds[1 - dirIsNeg[1]].y - ray.o.y) * invDir.y;

    // bounds intersection    
    if (tMin > tyMax || tyMin > tMax) return false;
    if (tyMin > tMin) tMin = tyMin;
    if (tyMax < tMax) tMax = tyMax;

    // Check for ray intersection against $z$ slab
    Float tzMin = (bounds[dirIsNeg[2]].z - ray.o.z) * invDir.z;
    Float tzMax = (bounds[1 - dirIsNeg[2]].z - ray.o.z) * invDir.z;

    // bounds intersection
    if (tMin > tzMax || tzMin > tMax) return false;
    if (tzMin > tMin) tMin = tzMin;
    if (tzMax < tMax) tMax = tzMax;
    return (tMin < ray.tMax) && (tMax > 0);
}

#endif // !__VECTOR_H