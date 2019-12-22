#pragma once
#ifndef __TRANSFORM_H
#define __TRANSFORM_H

#include "renderer/core/geometry.h"

class Matrix4x4 {
public:
    __host__ __device__ Matrix4x4() { memset(m, 0, sizeof(m)); }
    __host__ __device__ Matrix4x4(const Float mat[4][4]);
    __host__ __device__ Matrix4x4(const Float m[16]);
    __host__ __device__ Matrix4x4(
        Float m00, Float m01, Float m02, Float m03,
        Float m10, Float m11, Float m12, Float m13,
        Float m20, Float m21, Float m22, Float m23,
        Float m30, Float m31, Float m32, Float m33);

    __host__ __device__ void Identity();

    __host__ __device__ Matrix4x4 operator * (const Matrix4x4& t) const;

    Float m[4][4];

    friend __host__ __device__ Matrix4x4 Inverse(const Matrix4x4& t);
};

class Transform {
public:
    __host__ __device__ Transform() {}
    __host__ __device__ Transform(const Float m[16]);
    __host__ __device__ Transform(const Matrix4x4& m);
    __host__ __device__ Transform(const Matrix4x4& m, const Matrix4x4& invM);

    __host__ __device__ void Identity();

    __host__ __device__ Transform operator * (const Transform& t) const;
    __host__ __device__ Transform& operator *= (const Transform& t);

    template<typename T>
    __host__ __device__ Vector3<T> operator ()(const Vector3<T>& v) const;
    template<typename T>
    __host__ __device__ Point3<T> operator ()(const Point3<T>& p) const;
    template<typename T>
    __host__ __device__ Normal3<T> operator ()(const Normal3<T>& p) const;

    __host__ __device__ Ray operator() (const Ray& r) const;

    Matrix4x4 mat, invMat;

    friend __host__ __device__ Transform Inverse(const Transform& t);
};

__host__ __device__ inline
Transform Scale(Float x, Float y, Float z);
__host__ __device__ inline
Transform Translate(Float x, Float y, Float z);
__host__ __device__ inline
Transform Perspective(Float fov, Float near, Float far);


inline __device__ __host__
Matrix4x4::Matrix4x4(const Float mat[4][4]) 
{
    memcpy(m, mat, 16 * sizeof(Float));
}


inline __device__ __host__
Matrix4x4::Matrix4x4(const Float mat[16])
{
    memcpy(m, mat, 16 * sizeof(Float));
}

inline __host__ __device__ 
Matrix4x4::Matrix4x4(
    Float m00, Float m01, Float m02, Float m03,
    Float m10, Float m11, Float m12, Float m13,
    Float m20, Float m21, Float m22, Float m23,
    Float m30, Float m31, Float m32, Float m33) 
{
    m[0][0] = m00; m[0][1] = m01; m[0][2] = m02; m[0][3] = m03;
    m[1][0] = m10; m[1][1] = m11; m[1][2] = m12; m[1][3] = m13;
    m[2][0] = m20; m[2][1] = m21; m[2][2] = m22; m[2][3] = m23;
    m[3][0] = m30; m[3][1] = m31; m[3][2] = m32; m[3][3] = m33;
}

inline __device__ __host__
void Matrix4x4::Identity()
{
    memset(m, 0, sizeof(m));
    for (int i = 0; i < 4; i++) {
        m[i][i] = 1;
    }
}

inline __device__ __host__
Matrix4x4 Matrix4x4::operator*(const Matrix4x4& t) const
{
    Matrix4x4 ret;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            ret.m[i][j] = m[i][0] * t.m[0][j] +
            m[i][1] * t.m[1][j] +
            m[i][2] * t.m[2][j] +
            m[i][3] * t.m[3][j];
    return ret;
}

inline __device__ __host__
Matrix4x4 Inverse(const Matrix4x4& t)
{
    int indxc[4], indxr[4];
    int ipiv[4] = { 0, 0, 0, 0 };
    Float minv[4][4];
    memcpy(minv, t.m, 4 * 4 * sizeof(Float));
    for (int i = 0; i < 4; i++) {        
        int irow = 0, icol = 0;
        Float big = 0.f;
        // Choose pivot
        for (int j = 0; j < 4; j++) {
            if (ipiv[j] != 1) {
                for (int k = 0; k < 4; k++) {
                    if (ipiv[k] == 0) {
                        if (std::abs(minv[j][k]) >= big) {
                            big = Float(std::abs(minv[j][k]));
                            irow = j;
                            icol = k;
                        }
                    }
                    else {
                        ASSERT(ipiv[k] <= 1, "Singular matrix in MatrixInvert at Position 1");
                    }
                }
            }
        }
        ++ipiv[icol];
        // Swap rows _irow_ and _icol_ for pivot
        if (irow != icol) {
            for (int k = 0; k < 4; ++k)
                std::swap(minv[irow][k], minv[icol][k]);
        }
        indxr[i] = irow;
        indxc[i] = icol;

        ASSERT(minv[icol][icol] != 0.f, "Singular matrix in MatrixInvert at Position 2");

        // Set $m[icol][icol]$ to one by scaling row _icol_ appropriately
        Float pivinv = Float(1) / minv[icol][icol];
        minv[icol][icol] = 1.;
        for (int j = 0; j < 4; j++)
            minv[icol][j] *= pivinv;

        // Subtract this row from others to zero out their columns
        for (int j = 0; j < 4; j++) {
            if (j != icol) {
                Float save = minv[j][icol];
                minv[j][icol] = 0;
                for (int k = 0; k < 4; k++)
                    minv[j][k] -= minv[icol][k] * save;
            }
        }
    }
    // Swap columns to reflect permutation
    for (int j = 3; j >= 0; j--) {
        if (indxr[j] != indxc[j]) {
            for (int k = 0; k < 4; k++)
                std::swap(minv[k][indxr[j]], minv[k][indxc[j]]);
        }
    }
    return Matrix4x4(minv);
}

inline __device__ __host__
Transform::Transform(const Float m[16])
{
    mat = Matrix4x4(m);
    invMat = Inverse(mat);
}

inline __host__ __device__
Transform::Transform(const Matrix4x4& m) :mat(m), invMat(Inverse(m))
{
}

inline __host__ __device__
Transform::Transform(const Matrix4x4& m, const Matrix4x4& invM) :mat(m), invMat(invM)
{
}

inline __device__ __host__
void Transform::Identity()
{
    mat.Identity();
    invMat.Identity();
}

inline __device__ __host__
Transform Transform::operator*(const Transform& t) const
{
    return Transform(mat * t.mat, t.invMat * invMat);
}

inline __device__ __host__
Transform& Transform::operator*=(const Transform& t)
{
    mat = mat * t.mat;
    invMat = t.invMat * invMat;
    return *this;
}

template<typename T>
inline __device__ __host__
Vector3<T> Transform::operator()(const Vector3<T>& v) const
{
    return Vector3<T>(
        mat.m[0][0] * v.x + mat.m[0][1] * v.y + mat.m[0][2] * v.z,
        mat.m[1][0] * v.x + mat.m[1][1] * v.y + mat.m[1][2] * v.z,
        mat.m[2][0] * v.x + mat.m[2][1] * v.y + mat.m[2][2] * v.z);

}

template<typename T>
inline __device__ __host__
Point3<T> Transform::operator()(const Point3<T>& p) const
{
    T x = p.x, y = p.y, z = p.z;
    T xp = mat.m[0][0] * x + mat.m[0][1] * y + mat.m[0][2] * z + mat.m[0][3];
    T yp = mat.m[1][0] * x + mat.m[1][1] * y + mat.m[1][2] * z + mat.m[1][3];
    T zp = mat.m[2][0] * x + mat.m[2][1] * y + mat.m[2][2] * z + mat.m[2][3];
    T wp = mat.m[3][0] * x + mat.m[3][1] * y + mat.m[3][2] * z + mat.m[3][3];
    ASSERT(wp != 0, "Divide Zero");
    if (wp == 1)
        return Point3<T>(xp, yp, zp);
    else
        return Point3<T>(xp, yp, zp) / wp;
}

template<typename T>
inline __device__ __host__
Normal3<T> Transform::operator()(const Normal3<T>& n) const
{
    return Normal3<T>(
        invMat.m[0][0] * n.x + invMat.m[1][0] * n.y + invMat.m[2][0] * n.z,
        invMat.m[0][1] * n.x + invMat.m[1][1] * n.y + invMat.m[2][1] * n.z,
        invMat.m[0][2] * n.x + invMat.m[1][2] * n.y + invMat.m[2][2] * n.z);
}

inline __device__ __host__
Ray Transform::operator()(const Ray& r) const
{
    return Ray((*this)(r.o), (*this)(r.d), r.tMax);
}

inline __device__ __host__
Transform Inverse(const Transform& t)
{
    return Transform(t.invMat, t.mat);
}

inline __device__ __host__
Transform Scale(Float x, Float y, Float z)
{
    Matrix4x4 m(x, 0, 0, 0,
        0, y, 0, 0,
        0, 0, z, 0,
        0, 0, 0, 1);
    Matrix4x4 mInv(1.f / x, 0, 0, 0,
        0, 1.f / y, 0, 0,
        0, 0, 1.f / z, 0,
        0, 0, 0, 1.f);
    return Transform(m, mInv);
}

inline __device__ __host__
Transform Translate(Float x, Float y, Float z)
{
    Matrix4x4 m(1, 0, 0, x,
        0, 1, 0, y,
        0, 0, 1, z,
        0, 0, 0, 1);
    Matrix4x4 mInv(1, 0, 0, -x,
        0, 1, 0, -y,
        0, 0, 1, -z,
        0, 0, 0, 1);
    return Transform(m, mInv);
}

inline __host__ __device__ 
Transform Perspective(Float fov, Float n, Float f)
{
    Float invTanAng = 1. / std::tan(Radians(fov * 0.5f));
    Matrix4x4 perspective(
        invTanAng, 0, 0, 0,
        0, invTanAng, 0, 0,
        0, 0, f / (f - n), -f * n / (f - n), 
        0, 0, 1, 0);
    return Transform(perspective);
}



#endif // __TRANSFORM_H