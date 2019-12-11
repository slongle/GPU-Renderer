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


Transform Inverse(const Transform& t)
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
