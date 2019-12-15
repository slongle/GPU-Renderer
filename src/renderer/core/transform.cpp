#include "transform.h"

Matrix4x4 Inverse(const Matrix4x4& t)
{
    return Matrix4x4();
}

Transform::Transform(const Float m[16])
{
}

void Transform::Identity()
{
}

Transform Transform::operator*(const Transform& t) const
{
    return Transform();
}

Transform& Transform::operator*=(const Transform& t)
{
    return *this;
}

Ray Transform::operator()(const Ray& r) const
{
    return Ray((*this)(r.o), (*this)(r.d), r.tMax);
}

Transform Inverse(const Transform& t)
{
    return Transform();
}

Transform Scale(Float x, Float y, Float z)
{
    return Transform();
}

Transform Translate(Float x, Float y, Float z)
{
    return Transform();
}

Transform Perspective(Float fov, Float near, Float far)
{
    return Transform();
}

Matrix4x4::Matrix4x4(const Float m[16])
{
}

void Matrix4x4::Identity()
{
}

Matrix4x4 Matrix4x4::operator*(const Matrix4x4& t) const
{
    return Matrix4x4();
}

Matrix4x4& Matrix4x4::operator*=(const Matrix4x4& t)
{
    return *this;
    // TODO: insert return statement here
}
