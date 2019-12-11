#pragma once
#ifndef __TRIANGLEMESH_H
#define __TRIANGLEMESH_H

#include "renderer/core/shape.h"

class TriangleMesh{

};

class Triangle :public Shape {

};

std::vector<std::shared_ptr<Shape>>
CreateTriangleMeshShape(
    const ParameterSet& params, 
    Transform objToWorld, 
    Transform worldToObj);

#endif // !__TRIANGLEMESH_H
