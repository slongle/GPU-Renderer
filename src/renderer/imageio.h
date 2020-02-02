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
    Buffer<HOST_BUFFER, uint8>& buffer);

void ReadImage(
    const std::string& filename,
    int* width,
    int* height,
    Buffer<HOST_BUFFER, float>& buffer);

void WriteImage(
    const std::string& filename,
    const int& width, 
    const int& height,
    uint8* buffer);