#pragma once
#ifndef __TRANSFORM_H
#define __TRANSFORM_H

#include "renderer/core/fwd.h"
#include "renderer/core/geometry.h"

class Matrix4x4 {
public:
    Matrix4x4() {}
    Matrix4x4(const Float m[16]);

    void Identity();

    Matrix4x4 operator * (const Matrix4x4& t) const;
    Matrix4x4& operator *= (const Matrix4x4& t);

    Float m[16];

    friend Matrix4x4 Inverse(const Matrix4x4& t);
};

class Transform {
public:
    Transform() {}
    Transform(const Float m[16]);

    void Identity();

    Transform operator * (const Transform& t) const;
    Transform& operator *= (const Transform& t);

    template<typename T>
    Vector3<T> operator ()(const Vector3<T>& v) const;
    template<typename T>
    Point3<T> operator ()(const Point3<T>& p) const;
    template<typename T>
    Normal3<T> operator ()(const Normal3<T>& p) const;

    Ray operator() (const Ray& r) const;

    Matrix4x4 mat, invMat;

    friend Transform Inverse(const Transform& t);
};

Transform Scale(Float x, Float y, Float z);
Transform Translate(Float x, Float y, Float z);
Transform Perspective(Float fov, Float near, Float far);

template<typename T>
inline Vector3<T> Transform::operator()(const Vector3<T>& v) const
{
    return Vector3<T>(
        mat.m[0] * v.x + mat.m[1] * v.y + mat.m[2] * v.z,
        mat.m[4] * v.x + mat.m[5] * v.y + mat.m[6] * v.z,
        mat.m[8] * v.x + mat.m[9] * v.y + mat.m[10] * v.z);
}

template<typename T>
inline Point3<T> Transform::operator()(const Point3<T>& p) const
{
    T x =  mat.m[0] * p.x +  mat.m[1] * p.y + mat.m[2]  * p.z + mat.m[3];
    T y =  mat.m[4] * p.x +  mat.m[5] * p.y + mat.m[6]  * p.z + mat.m[7];
    T z =  mat.m[8] * p.x +  mat.m[9] * p.y + mat.m[10] * p.z + mat.m[11];
    T w = mat.m[12] * p.x + mat.m[13] * p.y + mat.m[14] * p.z + mat.m[15];
    ASSERT(w != 0, "Divide Zero");
    if (w == 1)
        return Point3<T>(x, y, z);
    else
        return Point3<T>(x, y, z) / w;
}

template<typename T>
inline Normal3<T> Transform::operator()(const Normal3<T>& n) const
{
    return Normal3<T>(
        invMat.m[0] * n.x + invMat.m[1] * n.y + invMat.m[2]  * n.z,
        invMat.m[4] * n.x + invMat.m[5] * n.y + invMat.m[6]  * n.z,
        invMat.m[8] * n.x + invMat.m[9] * n.y + invMat.m[10] * n.z);
}

#endif // __TRANSFORM_H