#pragma once

#include "renderer/fwd.h"

class Matrix4x4 {
public:
    Matrix4x4() {
        m[0][0] = m[1][1] = m[2][2] = m[3][3] = 1;
        m[0][1] = m[0][2] = m[0][3] = m[1][0] = m[1][2] = m[1][3] =
            m[2][0] = m[2][1] = m[2][3] = m[3][0] = m[3][1] = m[3][2] = 0;
    }

    Matrix4x4(const float mat[4][4]);
    Matrix4x4(float m00, float m01, float m02, float m03,
        float m10, float m11, float m12, float m13,
        float m20, float m21, float m22, float m23,
        float m30, float m31, float m32, float m33);

    void Identify();

    bool operator == (const Matrix4x4& m1) const {
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                if (m[i][j] != m1.m[i][j])
                    return false;
        return true;
    }

    bool operator != (const Matrix4x4& m1) const {
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                if (m[i][j] != m1.m[i][j])
                    return true;
        return false;
    }

    float& operator [](const int index) {
        return m[index / 4][index % 4];
    }

    Matrix4x4 operator * (const Matrix4x4& m1) const {
        Matrix4x4 ret;
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                ret.m[i][j] = m[i][0] * m1.m[0][j] +
                m[i][1] * m1.m[1][j] +
                m[i][2] * m1.m[2][j] +
                m[i][3] * m1.m[3][j];
        return ret;
    }

    friend Matrix4x4 Transpose(const Matrix4x4& m1);
    friend Matrix4x4 Inverse(const Matrix4x4& m1);

    friend std::istream& operator >>(std::istream& os, Matrix4x4& m1) {
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                os >> m1.m[i][j];
        return os;
    }

    float m[4][4];
};

Matrix4x4 toMatrix(const std::string& str);

class Transform {
public:
    Transform() { Identify(); }
    Transform(const float mat[4][4]) {
        m = Matrix4x4(mat[0][0], mat[0][1], mat[0][2], mat[0][3],
            mat[1][0], mat[1][1], mat[1][2], mat[1][3],
            mat[2][0], mat[2][1], mat[2][2], mat[2][3],
            mat[3][0], mat[3][1], mat[3][2], mat[3][3]);
        mInv = Inverse(m);
    }
    Transform(const Matrix4x4& _m) :m(_m), mInv(Inverse(m)) {}
    Transform(const Matrix4x4& _m, const Matrix4x4& _mInv) :m(_m), mInv(_mInv) {}

    void Identify();

    bool operator == (const Transform& t) const {
        return m == t.m;
    }

    Transform operator * (const Transform& t) const {
        return Transform(m * t.m, t.mInv * mInv);
    }

    Transform& operator *= (const Transform& t) {
        m = m * t.m;
        mInv = t.mInv * mInv;
        return *this;
    }
    
    float3 operator () (const float3& p) const;
    float3 transformPoint(const float3& p) const;
    float3 transformVector(const float3& p) const;
    float3 transformNormal(const float3& p) const;

    friend Transform Inverse(const Transform& t) {
        return Transform(t.mInv, t.m);
    }

    friend Transform Transpose(const Transform& t) {
        return Transform(Transpose(t.m), Transpose(t.mInv));
    }

    const static Transform identityTransform;

private:
    Matrix4x4 m, mInv;
};

Transform Translate(const float& x, const float& y, const float& z);
Transform Translate(const float3& delta);
Transform Scale(const float3& scale);
Transform Scale(float x, float y, float z);
Transform RotateX(float theta);
Transform RotateY(float theta);
Transform RotateZ(float theta);
Transform Rotate(float theta, const float3& axis);
Transform LookAt(const float3& target, const float3& origin, const float3& up);
Transform Perspective(const float& fov, const float& near, const float& far);

inline HOST_DEVICE
float3 Transform::transformPoint(const float3& p) const
{
    float x = m.m[0][0] * p.x + m.m[0][1] * p.y + m.m[0][2] * p.z + m.m[0][3];
    float y = m.m[1][0] * p.x + m.m[1][1] * p.y + m.m[1][2] * p.z + m.m[1][3];
    float z = m.m[2][0] * p.x + m.m[2][1] * p.y + m.m[2][2] * p.z + m.m[2][3];
    float w = m.m[3][0] * p.x + m.m[3][1] * p.y + m.m[3][2] * p.z + m.m[3][3];
    assert(w != 0, "Divide Zero");
    if (w == 1)
        return make_float3(x, y, z);
    else
        return make_float3(x, y, z) / w;
}

inline HOST_DEVICE
float3 Transform::transformVector(const float3& v) const
{
    float x = m.m[0][0] * v.x + m.m[0][1] * v.y + m.m[0][2] * v.z + m.m[0][3];
    float y = m.m[1][0] * v.x + m.m[1][1] * v.y + m.m[1][2] * v.z + m.m[1][3];
    float z = m.m[2][0] * v.x + m.m[2][1] * v.y + m.m[2][2] * v.z + m.m[2][3];
    return make_float3(x, y, z);
}

inline HOST_DEVICE
float3 Transform::transformNormal(const float3& n) const
{
    float x = mInv.m[0][0] * n.x + mInv.m[1][0] * n.y + mInv.m[2][0] * n.z;
    float y = mInv.m[0][1] * n.x + mInv.m[1][1] * n.y + mInv.m[2][1] * n.z;
    float z = mInv.m[0][2] * n.x + mInv.m[1][2] * n.y + mInv.m[2][2] * n.z;
    return make_float3(x, y, z);
}
