#pragma once

#include "renderer/fwd.h"

void ReadImage(
    const std::string& filename,
    int* width, 
    int* height,
    uint8*& buffer);

void ReadImage(
    const std::string& filename,
    int* width,
    int* height,
    float*& buffer);

void ReadImage(
    const std::string& filename,
    int* width,
    int* height,
    std::vector<uint8>& buffer);

void ReadImage(
    const std::string& filename,
    int* width,
    int* height,
    std::vector<float>& buffer);

void WriteImage(
    const std::string& filename,
    const int& width, 
    const int& height,
    uint8* buffer);