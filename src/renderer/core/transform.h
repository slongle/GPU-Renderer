#pragma once
#ifndef __TRANSFORM_H
#define __TRANSFORM_H

#include "renderer/core/fwd.h"

class Matrix4x4 {
public:

    Float m[16];
};

class Transform {
public:

    Matrix4x4 mat, invMat;
};

#endif // __TRANSFORM_H
