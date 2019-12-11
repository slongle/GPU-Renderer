#pragma once
#ifndef __CAMERA_H
#define __CAMERA_H

#include "renderer/core/fwd.h"
#include "renderer/core/transform.h"
#include "renderer/core/parameterset.h"

class Camera {

};

std::shared_ptr<Camera>
CreateCamera(
    const ParameterSet& param,
    const Transform objToWorld,
    const Transform worldToObj);

#endif // !__CAMERA_H
