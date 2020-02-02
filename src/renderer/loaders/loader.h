#pragma once

#include <vector>

#include "renderer/fwd.h"
#include "renderer/triangle.h"
#include "renderer/camera.h"
#include "renderer/loaders/objloader.h"
#include "renderer/loaders/mitsubaloader.h"

inline 
void load_scene(
    const std::string& filename, 
    Scene* scene)
{
    std::string ext = getFileExtension(filename);
    if (ext == ".obj")
    {
        load_obj_mtl_file(filename, scene);
    }
    else if (ext == ".xml")
    {
        load_mitsuba_file(filename, scene);
    }
    else
    {
        assert(false);
    }
}