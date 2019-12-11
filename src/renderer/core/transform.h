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

    Float m_m[16];

    friend Matrix4x4 Inverse(const Matrix4x4& t);
};

class Transform {
public:
    Transform() {}
    Transform(const Float m[16]);
    Transform(const Matrix4x4 m);
    Transform(const Float m[16],const Float mInv[16]);
    Transform(const Matrix4x4 m, const Matrix4x4 mInv);
    Transform(const Matrix4x4 m, const Float mInv[16]);
    Transform(const Float m[16], const Matrix4x4 mInv);

    void Identity();

    Transform operator * (const Transform& t) const;
    Transform& operator *= (const Transform& t);
    

    Matrix4x4 m_mat, m_invMat;

    friend Transform Inverse(const Transform& t);
};

#endif // __TRANSFORM_H
