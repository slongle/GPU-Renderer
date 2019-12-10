#pragma once
#ifndef __TRANSFORM_H
#define __TRANSFORM_H

#include "renderer/core/fwd.h"

class Matrix4x4 {
public:
    Matrix4x4() {}
    Matrix4x4(const Float m[16]);

    void Identity();

    Matrix4x4 operator * (const Matrix4x4& t) const;
    Matrix4x4& operator *= (const Matrix4x4& t);

    Float m[16];
};

class Transform {
public:
    Transform() {}
    Transform(const Float m[16]);

    void Identity();

    Transform operator * (const Transform& t) const;
    Transform& operator *= (const Transform& t);

    Matrix4x4 mat, invMat;
};

#endif // __TRANSFORM_H
