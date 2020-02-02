#pragma once
#include <string>
#include <vector>

#include "renderer/triangle.h"

bool load_obj_mtl_file(const std::string& filename, Scene* scene);
bool load_obj_file(const std::string& filename, std::vector<Triangle>& triangles);