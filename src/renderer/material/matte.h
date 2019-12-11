#pragma once
#ifndef __MATTE_H
#define __MATTE_H

#include "renderer/core/material.h"

class Matte :public Material {
public:

};

Matte* CreateMatteMaterial(const ParameterSet& param);

#endif // !__MATTE_H
