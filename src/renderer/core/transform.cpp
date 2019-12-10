#include "transform.h"

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
