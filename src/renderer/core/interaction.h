#pragma once
#ifndef __INTERACTION_H
#define __INTERACTION_H

#include "renderer/core/fwd.h"

class Interaction {
public:
    __host__ __device__
    Interaction() {}

    int m_primitiveID;
};

#endif // !__INTERACTION_H
