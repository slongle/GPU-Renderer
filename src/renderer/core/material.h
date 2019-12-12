#pragma once
#ifndef __MATERIAL_H
#define __MATERIAL_H

#include "renderer/core/fwd.h"
#include "renderer/core/parameterset.h"

class Material {
public:

};

std::shared_ptr<Material>
CreateMatteMaterial(const ParameterSet& param);

#endif // !__MATERIAL_H
