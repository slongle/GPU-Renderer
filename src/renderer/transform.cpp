#include "transform.h"

Matrix4x4::Matrix4x4(const float mat[4][4]) {
    memcpy(m, mat, 16 * sizeof(float));
}

Matrix4x4::Matrix4x4(float m00, float m01, float m02, float m03,
    float m10, float m11, float m12, float m13,
    float m20, float m21, float m22, float m23,
    float m30, float m31, float m32, float m33) {
    m[0][0] = m00; m[0][1] = m01; m[0][2] = m02; m[0][3] = m03;
    m[1][0] = m10; m[1][1] = m11; m[1][2] = m12; m[1][3] = m13;
    m[2][0] = m20; m[2][1] = m21; m[2][2] = m22; m[2][3] = m23;
    m[3][0] = m30; m[3][1] = m31; m[3][2] = m32; m[3][3] = m33;
}

void Matrix4x4::Identify() {
    m[0][0] = m[1][1] = m[2][2] = m[3][3] = 1;
    m[0][1] = m[0][2] = m[0][3] = m[1][0] = m[1][2] = m[1][3] =
        m[2][0] = m[2][1] = m[2][3] = m[3][0] = m[3][1] = m[3][2] = 0;
}

Matrix4x4 Transpose(const Matrix4x4& m1) {
    return Matrix4x4(m1.m[0][0], m1.m[1][0], m1.m[2][0], m1.m[3][0],
        m1.m[0][1], m1.m[1][1], m1.m[2][1], m1.m[3][1],
        m1.m[0][2], m1.m[1][2], m1.m[2][2], m1.m[3][2],
        m1.m[0][3], m1.m[1][3], m1.m[2][3], m1.m[3][3]);
}

Matrix4x4 Inverse(const Matrix4x4& m1) {
    int indxc[4], indxr[4];
    int ipiv[4] = { 0, 0, 0, 0 };
    float minv[4][4];
    memcpy(minv, m1.m, 4 * 4 * sizeof(float));
    for (int i = 0; i < 4; i++) {
        int irow = 0, icol = 0;
        float big = 0.f;
        // Choose pivot
        for (int j = 0; j < 4; j++) {
            if (ipiv[j] != 1) {
                for (int k = 0; k < 4; k++) {
                    if (ipiv[k] == 0) {
                        if (std::abs(minv[j][k]) >= big) {
                            big = float(std::abs(minv[j][k]));
                            irow = j;
                            icol = k;
                        }
                    }
                    else {
                        assert(ipiv[k] <= 1, "Singular matrix in MatrixInvert at Position 1");
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

        assert(minv[icol][icol] != 0.f, "Singular matrix in MatrixInvert at Position 2");

        // Set $m[icol][icol]$ to one by scaling row _icol_ appropriately
        float pivinv = float(1) / minv[icol][icol];
        minv[icol][icol] = 1.;
        for (int j = 0; j < 4; j++)
            minv[icol][j] *= pivinv;

        // Subtract this row from others to zero out their columns
        for (int j = 0; j < 4; j++) {
            if (j != icol) {
                float save = minv[j][icol];
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

Matrix4x4 toMatrix(const std::string& str) {
    Matrix4x4 ret;
    char* endptr;
    ret[0] = strtof(str.c_str(), &endptr);
    for (int i = 1; i < 16; i++) {
        endptr++;
        ret[i] = strtof(endptr, &endptr);
    }
    return ret;
}

const Transform Transform::identityTransform = Transform();

Transform Translate(const float& x, const float& y, const float& z) {
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

Transform Translate(const float3& delta) {
    Matrix4x4 m(1, 0, 0, delta.x,
        0, 1, 0, delta.y,
        0, 0, 1, delta.z,
        0, 0, 0, 1);
    Matrix4x4 mInv(1, 0, 0, -delta.x,
        0, 1, 0, -delta.y,
        0, 0, 1, -delta.z,
        0, 0, 0, 1);
    return Transform(m, mInv);
}

Transform Scale(const float3& scale) {
    Matrix4x4 m(scale.x, 0, 0, 0,
        0, scale.y, 0, 0,
        0, 0, scale.z, 0,
        0, 0, 0, 1);
    Matrix4x4 mInv(1. / scale.x, 0, 0, 0,
        0, 1. / scale.y, 0, 0,
        0, 0, 1. / scale.z, 0,
        0, 0, 0, 1);
    return Transform(m, mInv);
}

Transform Scale(float x, float y, float z) {
    Matrix4x4 m(x, 0, 0, 0,
        0, y, 0, 0,
        0, 0, z, 0,
        0, 0, 0, 1);
    Matrix4x4 mInv(1. / x, 0, 0, 0,
        0, 1. / y, 0, 0,
        0, 0, 1. / z, 0,
        0, 0, 0, 1.);
    return Transform(m, mInv);
}

Transform RotateX(float theta) {
    float sinTheta = std::sin(degToRad(theta));
    float cosTheta = std::cos(degToRad(theta));
    Matrix4x4 m(1, 0, 0, 0,
        0, cosTheta, -sinTheta, 0,
        0, sinTheta, cosTheta, 0,
        0, 0, 0, 1);
    Matrix4x4 mInv(1, 0, 0, 0,
        0, cosTheta, sinTheta, 0,
        0, -sinTheta, cosTheta, 0,
        0, 0, 0, 1);
    return Transform(m, mInv);
}

Transform RotateY(float theta) {
    float sinTheta = std::sin(degToRad(theta));
    float cosTheta = std::cos(degToRad(theta));
    Matrix4x4 m(cosTheta, 0, sinTheta, 0,
        0, 1, 0, 0,
        -sinTheta, 0, cosTheta, 0,
        0, 0, 0, 1);
    Matrix4x4 mInv(cosTheta, 0, -sinTheta, 0,
        0, 1, 0, 0,
        sinTheta, 0, cosTheta, 0,
        0, 0, 0, 1);
    return Transform(m, mInv);
}

Transform RotateZ(float theta) {
    float sinTheta = std::sin(degToRad(theta));
    float cosTheta = std::cos(degToRad(theta));
    Matrix4x4 m(cosTheta, -sinTheta, 0, 0,
        sinTheta, cosTheta, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1);
    Matrix4x4 mInv(1, 0, 0, 0,
        0, cosTheta, sinTheta, 0,
        0, -sinTheta, cosTheta, 0,
        0, 0, 0, 1);
    return Transform(m, mInv);
}

Transform Rotate(float theta, const float3& axis) {
    float3 a = normalize(axis);
    float sinTheta = std::sin(degToRad(theta));
    float cosTheta = std::cos(degToRad(theta));
    Matrix4x4 m;
    // Compute rotation of first basis vector
    m.m[0][0] = a.x * a.x + (1 - a.x * a.x) * cosTheta;
    m.m[0][1] = a.x * a.y * (1 - cosTheta) - a.z * sinTheta;
    m.m[0][2] = a.x * a.z * (1 - cosTheta) + a.y * sinTheta;
    m.m[0][3] = 0;

    // Compute rotations of second basis vector
    m.m[1][0] = a.x * a.y * (1 - cosTheta) + a.z * sinTheta;
    m.m[1][1] = a.y * a.y + (1 - a.y * a.y) * cosTheta;
    m.m[1][2] = a.y * a.z * (1 - cosTheta) - a.x * sinTheta;
    m.m[1][3] = 0;

    // Compute rotations of third basis vector
    m.m[2][0] = a.x * a.z * (1 - cosTheta) - a.y * sinTheta;
    m.m[2][1] = a.y * a.z * (1 - cosTheta) + a.x * sinTheta;
    m.m[2][2] = a.z * a.z + (1 - a.z * a.z) * cosTheta;
    m.m[2][3] = 0;
    return Transform(m, Transpose(m));
}

Transform LookAt(const float3& to, const float3& from, const float3& up) {
    Matrix4x4 cameraToWorld;
    cameraToWorld.m[0][3] = from.x;
    cameraToWorld.m[1][3] = from.y;
    cameraToWorld.m[2][3] = from.z;
    cameraToWorld.m[3][3] = 1;

    float3 forward = normalize(to - from);
    float3 left = normalize(cross(up, forward));
    float3 realUp = cross(forward, left);
    cameraToWorld.m[0][0] = left.x;
    cameraToWorld.m[1][0] = left.y;
    cameraToWorld.m[2][0] = left.z;
    cameraToWorld.m[3][0] = 0;

    cameraToWorld.m[0][1] = realUp.x;
    cameraToWorld.m[1][1] = realUp.y;
    cameraToWorld.m[2][1] = realUp.z;
    cameraToWorld.m[3][1] = 0;

    cameraToWorld.m[0][2] = forward.x;
    cameraToWorld.m[1][2] = forward.y;
    cameraToWorld.m[2][2] = forward.z;
    cameraToWorld.m[3][2] = 0;

    //std::cout << cameraToWorld << std::endl;

    return Transform(cameraToWorld, Inverse(cameraToWorld));
}

Transform Perspective(const float& fov, const float& n, const float& f) {
    float invTanAng = 1. / std::tan(degToRad(fov * 0.5f));
    Matrix4x4 perspective(
        invTanAng, 0, 0, 0,
        0, invTanAng, 0, 0,
        0, 0, f / (f - n), -f * n / (f - n),
        0, 0, 1, 0);
    return Transform(perspective);
}

float3 Transform::operator()(const float3& p) const {
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

void Transform::Identify() {
    m.Identify();
    mInv.Identify();
}
